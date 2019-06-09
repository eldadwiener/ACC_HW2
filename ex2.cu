/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <string.h>

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
#define IMG_DIMENSION 32
#define NREQUESTS 10000
#define DONEJOB -1
typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

void process_image(uchar *img_in, uchar *img_out) {
    int histogram[256] = { 0 };
    for (int i = 0; i < SQR(IMG_DIMENSION); i++) {
        histogram[img_in[i]]++;
    }

    int cdf[256] = { 0 };
    int hist_sum = 0;
    for (int i = 0; i < 256; i++) {
        hist_sum += histogram[i];
        cdf[i] = hist_sum;
    }

    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    uchar map[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        int map_value = (float)(cdf[i] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[i] = (uchar)map_value;
    }

    for (int i = 0; i < SQR(IMG_DIMENSION); i++) {
        img_out[i] = map[img_in[i]];
    }
}

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}

/* we'll use these to rate limit the request load */
struct rate_limit_t {
    double last_checked;
    double lambda;
    unsigned seed;
};


void rate_limit_init(struct rate_limit_t *rate_limit, double lambda, int seed) {
    rate_limit->lambda = lambda;
    rate_limit->seed = (seed == -1) ? 0 : seed;
    rate_limit->last_checked = 0;
}

int rate_limit_can_send(struct rate_limit_t *rate_limit) {
    if (rate_limit->lambda == 0) return 1;
    double now = get_time_msec() * 1e-3;
    double dt = now - rate_limit->last_checked;
    double p = dt * rate_limit->lambda;
    rate_limit->last_checked = now;
    if (p > 1) p = 1;
    double r = (double)rand_r(&rate_limit->seed) / RAND_MAX;
    return (p > r);
}

double distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    double distance_sqr = 0;
    for (int i = 0; i < NREQUESTS * SQR(IMG_DIMENSION); i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}

/* we won't load actual files. just fill the images with random bytes */
void load_images(uchar *images) {
    srand(0);
    for (int i = 0; i < NREQUESTS * SQR(IMG_DIMENSION); i++) {
        images[i] = rand() % 256;
    }
}

__device__ int arr_min(int arr[], int arr_size) {
    // we assume arr_size threads call this function for arr[]
    __shared__ int SharedMin;
    int tid = threadIdx.x;
    for(int stride = 0; stride < arr_size; stride += blockDim.x)
    {
        if( (tid + stride < arr_size) && 
            (arr[tid + stride] > 0) && 
            ((tid + stride == 0) || (arr[tid + stride - 1] == 0))) // cdf is a rising function, so only the first non zero will have zero before it.
        {
            SharedMin = arr[tid + stride];
        }
        __syncthreads();
    }
    return SharedMin;
}

__device__ int arr_min_ref(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int rhs, lhs;

    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            rhs = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            lhs = arr[tid];
            if (rhs != 0) {
                if (lhs == 0)
                    arr[tid] = rhs;
                else
                    arr[tid] = min(arr[tid], rhs);
            }
        }
        __syncthreads();
    }

    int ret = arr[arr_size - 1];
    return ret;
}

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;

    for (int stride = 1; stride < min(blockDim.x, arr_size); stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__global__ void gpu_process_image(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ int hist_min[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        hist_min[tid] = histogram[tid];
    }
    __syncthreads();

    int cdf_min = arr_min(hist_min, 256);

    __shared__ uchar map[256];
    if (tid < 256) {
        int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[tid] = (uchar)map_value;
    }

    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
        out[i] = map[in[i]];
    }
    return;
}

void print_usage_and_die(char *progname) {
    printf("usage:\n");
    printf("%s streams <load (requests/sec)>\n", progname);
    printf("OR\n");
    printf("%s queue <#threads> <load (requests/sec)>\n", progname);
    exit(1);
}

int _min(int a, int b) {
    return (a<b) ? a : b;
}

unsigned int getTBlocksAmnt(int threadsPerBlock, int shmemPerBlock) {
    struct cudaDeviceProp props;
    CUDA_CHECK( cudaGetDeviceProperties(&props, 0) );
    int  ThreadsPerSM = min(props.maxThreadsPerMultiProcessor, props.regsPerMultiprocessor/32);
    int  SMCount = props.multiProcessorCount;
    size_t  shmemPerSM = props.sharedMemPerMultiprocessor;
    return SMCount * min( ThreadsPerSM/threadsPerBlock, (unsigned int)shmemPerSM/shmemPerBlock);
}

#define QSIZE 10
typedef struct QmetaData {
    int tail;
    int head;
    int size;
    bool done;
} QmetaData;

typedef struct jobS* pJobS;
typedef struct jobS {
    uchar job[SQR(IMG_DIMENSION)];
    int jobId;
} jobS;

//TODO maybe remove?
typedef struct singleQ {
    jobS jobs[QSIZE];
    int head;
    int tail;
} Q;

typedef struct pcQ {
    QmetaData* meta;
    pJobS queue;
    int* usedCells;
}pcQ;

typedef struct tbMem {
    uchar memIn[SQR(IMG_DIMENSION)];
    uchar memOut[SQR(IMG_DIMENSION)];
} tbMem;

__host__ void setQ(pcQ& queue, void* allocated, unsigned int Qsize) {
    queue.meta = (QmetaData*)allocated;
    queue.meta->size = Qsize;
    queue.meta->head = 0;
    queue.meta->tail = 0;
    queue.meta->done = false;
    queue.queue = (pJobS)(queue.meta + 1);
    queue.usedCells = (int*)( queue.queue + queue.meta->size );
    for (int i = 0; i < queue.meta->size; ++i ){
        queue.usedCells[i] = 0;
    }
} 

__device__ void setQ(pcQ& queue, void* allocated) {
    queue.meta = (QmetaData*)allocated;
    queue.queue = (pJobS)(queue.meta + 1);
    queue.usedCells = (int*)( queue.queue + queue.meta->size );
}


__global__ void gpu_process_image_pc(void* in,void* out, tbMem* tb_mem) {
    __shared__ int histogram[256];
    __shared__ int hist_min[256];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // parse given pointers into useful structs
    pcQ inQ, outQ;
    setQ(inQ, in);
    setQ(outQ, out);
    tbMem currJob = tb_mem[bid];
    uchar * jobQptr;
    int* jobUsedCellPtr;
    unsigned int currJobId;
    int QbuffNum = inQ.meta->size;
    while (true){ 
        // if Q is empty and cpu is done, exit the kernel
        //try to catch a buffer
        __shared__ bool stop;// TODO add to threadblock amnt calculation
        stop = false;
        while (tid == 0){
            __threadfence();
            while(atomicCAS(&(inQ.usedCells[inQ.meta->tail % QbuffNum]), 1 ,2) != 1)
            {
                if(inQ.meta->tail == inQ.meta->head && inQ.meta->done) {
                    stop = true;
                    break;
                }
            }
            __threadfence();
            if(stop) break;
            int idx = inQ.meta->tail % QbuffNum;
            //save the job ptr and the job id
            jobQptr = inQ.queue[inQ.meta->tail % QbuffNum].job;
            currJobId = inQ.queue[inQ.meta->tail % QbuffNum].jobId;
            jobUsedCellPtr = inQ.usedCells + (inQ.meta->tail % QbuffNum);
            //move the tail forward to allow athoer T.Bs to work
            atomicAdd(&inQ.meta->tail,1);
            /* ---------TODO: do this copy mor efficient---------------------------------*/
            memcpy(currJob.memIn, jobQptr, SQR(IMG_DIMENSION));
            __threadfence();
            //*jobUsedCellPtr = 0; //the cell is empty now
            inQ.usedCells[idx] = 0;
            /*----------------------------------------------------------------------------*/
            break;
        }
        __syncthreads(); //wait for thread 0 to catch a job
        if (stop)
            return;
        /*do here the copy*/
        //
        //
        //do the calcs
        if (tid < 256) {
            histogram[tid] = 0;
        }
        __syncthreads();
    
        for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x)
            atomicAdd(&histogram[currJob.memIn[i]], 1);
    
        __syncthreads();
    
        prefix_sum(histogram, 256);
    
        if (tid < 256) {
            hist_min[tid] = histogram[tid];
        }
        __syncthreads();
    
        int cdf_min = arr_min(hist_min, 256);
    
        __shared__ uchar map[256];
        if (tid < 256) {
            int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
            map[tid] = (uchar)map_value;
        }
    
        __syncthreads();
    
        for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
            currJob.memOut[i] = map[currJob.memIn[i]];
        }
        // try to catch free cell in Qout and copy the result
        if (tid == 0){
            while(atomicCAS(&(outQ.usedCells[outQ.meta->head % QbuffNum]), 0 ,2) != 0);
            //save the job-out ptr and insert the job id
            jobQptr = outQ.queue[outQ.meta->head % QbuffNum].job;
            printf("GPU sending job #%d\n",outQ.queue[outQ.meta->head % QbuffNum].jobId); // TODO REMOVE
            outQ.queue[outQ.meta->head % QbuffNum].jobId = currJobId;
            jobUsedCellPtr = outQ.usedCells + (outQ.meta->head % QbuffNum);
            outQ.meta->head ++;
            /* ---------TODO: do this copy mor efficient---------------------------------*/
            memcpy(jobQptr, currJob.memOut, SQR(IMG_DIMENSION));
            *jobUsedCellPtr = 1; //the cell is ready for read now
            /*----------------------------------------------------------------------------*/
        }
    }
}


__global__ void gpu_process_image_pc2(volatile void* in,volatile void* out) {
    __shared__ int histogram[256];
    __shared__ int hist_min[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    uchar * jobQptr;
    int currJobId;
    Q *inQ = (Q*)in, *outQ = (Q*)out;
    inQ += bid;
    outQ += bid;
    while (true) {
        __threadfence();
        while(inQ->tail >= inQ->head) __threadfence_system(); // wait for queue to contain a job
        //save the job ptr and the job id
        jobQptr = inQ->jobs[inQ->tail % QSIZE].job;
        currJobId = inQ->jobs[inQ->tail % QSIZE].jobId;
        if(currJobId == DONEJOB) {
            __threadfence();
            return;
        }
        /*----------------------------------------------------------------------------*/
        /*do here the copy*/
        //
        //
        //do the calcs

        __threadfence();

        if (tid < 256) {
            histogram[tid] = 0;
        }
        __syncthreads();
    
        for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x)
        {
            __threadfence_system(); // needed to get currect image data
            atomicAdd(&histogram[jobQptr[i]], 1);
        }
    
        __syncthreads();
    
        prefix_sum(histogram, 256);
    
        if (tid < 256) {
            hist_min[tid] = histogram[tid];
        }
        __syncthreads();
    
        int cdf_min = arr_min(hist_min, 256);
    
        __shared__ uchar map[256];
        if (tid < 256) {
            int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
            map[tid] = (uchar)map_value;
        }
    
        __syncthreads();
        while(!(outQ->head - outQ->tail < QSIZE)) __threadfence_system();
        for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
            outQ->jobs[outQ->head % QSIZE].job[i] = map[jobQptr[i]];
        }
        outQ->jobs[outQ->head % QSIZE].jobId = currJobId; 
        __threadfence_system();
        // try to catch free cell in Qout and copy the result
        if (tid == 0){
            //save the job-out ptr and insert the job id
            outQ->head ++;
            inQ->tail ++;
            // printf("GPU: sent job #%d\n",currJobId);
        }
        __threadfence_system();
    }
}

enum {PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE};
int main(int argc, char *argv[]) {

    int mode = -1;
    int threads_queue_mode = -1; /* valid only when mode = queue */
    double load = 0;
    if (argc < 3) print_usage_and_die(argv[0]);

    if (!strcmp(argv[1], "streams")) {
        if (argc != 3) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_STREAMS;
        load = atof(argv[2]);
    } else if (!strcmp(argv[1], "queue")) {
        if (argc != 4) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_QUEUE;
        threads_queue_mode = atoi(argv[2]);
        load = atof(argv[3]);
    } else {
        print_usage_and_die(argv[0]);
    }

    uchar *images_in; /* we concatenate all images in one huge array */
    uchar *images_out;
    CUDA_CHECK( cudaHostAlloc(&images_in, NREQUESTS * SQR(IMG_DIMENSION), 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out, NREQUESTS * SQR(IMG_DIMENSION), 0) );

    load_images(images_in);
    double t_start, t_finish;

    /* using CPU */
    printf("\n=== CPU ===\n");
    t_start  = get_time_msec();
    for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx)
        process_image(&images_in[img_idx * SQR(IMG_DIMENSION)], &images_out[img_idx * SQR(IMG_DIMENSION)]);
    t_finish = get_time_msec();
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

    double total_distance = 0;

    /* using GPU task-serial.. just to verify the GPU code makes sense */
    printf("\n=== GPU Task Serial ===\n");

    uchar *images_out_from_gpu;
    CUDA_CHECK( cudaHostAlloc(&images_out_from_gpu, NREQUESTS * SQR(IMG_DIMENSION), 0) );

    do {
        uchar *gpu_image_in, *gpu_image_out;
        CUDA_CHECK(cudaMalloc(&gpu_image_in, SQR(IMG_DIMENSION)));
        CUDA_CHECK(cudaMalloc(&gpu_image_out, SQR(IMG_DIMENSION)));

        t_start = get_time_msec();
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
            CUDA_CHECK(cudaMemcpy(gpu_image_in, &images_in[img_idx * SQR(IMG_DIMENSION)], SQR(IMG_DIMENSION), cudaMemcpyHostToDevice));
            gpu_process_image<<<1, 1024>>>(gpu_image_in, gpu_image_out);
            CUDA_CHECK(cudaMemcpy(&images_out_from_gpu[img_idx * SQR(IMG_DIMENSION)], gpu_image_out, SQR(IMG_DIMENSION), cudaMemcpyDeviceToHost));
        }
        total_distance += distance_sqr_between_image_arrays(images_out, images_out_from_gpu);
        CUDA_CHECK(cudaDeviceSynchronize());
        t_finish = get_time_msec();
        printf("distance from baseline %lf (should be zero)\n", total_distance);
        printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

        CUDA_CHECK(cudaFree(gpu_image_in));
        CUDA_CHECK(cudaFree(gpu_image_out));
    } while (0);

    /* now for the client-server part */
    printf("\n=== Client-Server ===\n");
    double *req_t_start = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_start, 0, NREQUESTS * sizeof(double));

    double *req_t_end = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_end, 0, NREQUESTS * sizeof(double));

    struct rate_limit_t rate_limit;
    rate_limit_init(&rate_limit, load, 0);

    /* TODO allocate / initialize memory, streams, etc... */
    cudaStream_t streams[64];
    int imgInStream[64];
    uchar *gpu_image_in[64], *gpu_image_out[64];
    for(int i = 0; i < 64; i++) {
        cudaStreamCreate(&streams[i]);
        imgInStream[i] = -1;
        CUDA_CHECK(cudaMalloc(&gpu_image_in[i], SQR(IMG_DIMENSION)));
        CUDA_CHECK(cudaMalloc(&gpu_image_out[i], SQR(IMG_DIMENSION)));
    }
    CUDA_CHECK(cudaMemset(images_out_from_gpu, 0, NREQUESTS * SQR(IMG_DIMENSION)));

    double ti = get_time_msec();
    if (mode == PROGRAM_MODE_STREAMS) {
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {

            /* TODO query (don't block) streams for any completed requests.
             * update req_t_end of completed requests
             */
            int availStream = -1;
            for(int i=0; i < 64; ++i)
            {
                if(cudaStreamQuery(streams[i]) == cudaSuccess)
                {
                    if(availStream == -1)
                        availStream = i;
                    if(imgInStream[i] > -1)
                    {    
                        req_t_end[imgInStream[i]] = get_time_msec();
                        //printf("Img num: %d, start time: %f, end time: %f\n",imgInStream[i],req_t_start[imgInStream[i]],req_t_end[imgInStream[i]]); // REMOVE DEBUG
                        imgInStream[i] = -1;
                    }
                }
            }
            if (availStream == -1 || !rate_limit_can_send(&rate_limit)) {
                --img_idx;
                continue;
            }
            //printf("Sending img id: %d to stream id: %d\n",img_idx, availStream); // REMOVE DEBUG
            imgInStream[availStream] = img_idx;
            req_t_start[img_idx] = get_time_msec();

            /* TODO place memcpy's and kernels in a stream */
            cudaMemcpyAsync(gpu_image_in[availStream], &images_in[img_idx * SQR(IMG_DIMENSION)], SQR(IMG_DIMENSION),cudaMemcpyHostToDevice,streams[availStream]);
            gpu_process_image<<<1, SQR(IMG_DIMENSION) ,0, streams[availStream]>>>(gpu_image_in[availStream], gpu_image_out[availStream]);
            cudaMemcpyAsync(&images_out_from_gpu[img_idx * SQR(IMG_DIMENSION)], gpu_image_out[availStream], SQR(IMG_DIMENSION),cudaMemcpyDeviceToHost,streams[availStream]);
        }
        /* TODO now make sure to wait for all streams to finish */
        cudaDeviceSynchronize();
        // Mark all remaining images end time to now
        double endTime = get_time_msec();
        for(int i = 0; i < 64 ; ++i)
        {
            if(imgInStream[i] > -1)
            {
                req_t_end[imgInStream[i]] = endTime;
            }
        }
        //TODO, maybe need to move mem free further down
        for(int i = 0; i < 64; ++i)
        {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));    
        CUDA_CHECK(cudaFree(gpu_image_in[i]));
        CUDA_CHECK(cudaFree(gpu_image_out[i]));
        }

    } else if (mode == PROGRAM_MODE_QUEUE) {
        // TODO launch GPU consumer-producer kernel
        unsigned int tblocks = getTBlocksAmnt(threads_queue_mode, 2*4*256+256+4);
        unsigned int amntRecv = 0;
        unsigned int nextIns = 0; // first candidate block for next img insert, to implement RR
        Q *QinHost, *QinDev;
        Q *QoutHost, *QoutDev;
        CUDA_CHECK( cudaHostAlloc(&QinHost, sizeof(Q)*tblocks , 0) );
        CUDA_CHECK( cudaHostAlloc(&QoutHost, sizeof(Q)*tblocks , 0) );
        memset(QinHost, 0, sizeof(Q)*tblocks);
        memset(QoutHost, 0, sizeof(Q)*tblocks);
        CUDA_CHECK( cudaHostGetDevicePointer(&QinDev, QinHost, 0) );
        CUDA_CHECK( cudaHostGetDevicePointer(&QoutDev, QoutHost, 0) );
        gpu_process_image_pc2<<<tblocks, threads_queue_mode>>>(QinDev, QoutDev);
        __sync_synchronize();
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
            __sync_synchronize();
            // check each tblock's out queue
            for (int block = 0; block < tblocks; ++block)
            {
                __sync_synchronize();
                while ( QoutHost[block].tail < QoutHost[block].head )
                {
                    __sync_synchronize();
                    jobS& job = QoutHost[block].jobs[QoutHost[block].tail % QSIZE ];
                    // __sync_synchronize();
                    memcpy(images_out_from_gpu + (job.jobId * SQR(IMG_DIMENSION)), job.job, SQR(IMG_DIMENSION) );
                    // __sync_synchronize();
                    QoutHost[block].tail++;
                    __sync_synchronize();
                    req_t_end[job.jobId] = get_time_msec();
                    ++amntRecv;
                    // printf("received completed job #%d\n",job.jobId);
                }
            }
            // Check send rate limit
            if (!rate_limit_can_send(&rate_limit))
            {
                --img_idx;
                continue;
            }
            //printf("checking if theres room for img: %d\n",img_idx);
            int blockToUse;
            bool failed = true;
            for (int i = 0; i < tblocks; ++i)
            {
                blockToUse = (i + nextIns) % tblocks;
                //__sync_synchronize();
                if( QinHost[blockToUse].tail + QSIZE <= QinHost[blockToUse].head ) //the next cell isn't empty
                {
                    continue;
                } 
                //__sync_synchronize();
                failed = false;
                break;
            }
            if ( failed )
            {
                img_idx--;
                continue;
            }
            __sync_synchronize();
            req_t_start[img_idx] = get_time_msec();
            // TODO push task to queue 
            // printf("pushing img #%d, to threadblock #%d\n",img_idx, blockToUse);
            __sync_synchronize();
            jobS &inJob = QinHost[blockToUse].jobs[QinHost[blockToUse].head % QSIZE];
            __sync_synchronize();
            inJob.jobId = img_idx;
            __sync_synchronize();
            memcpy(inJob.job, images_in + (img_idx * SQR(IMG_DIMENSION)), SQR(IMG_DIMENSION));
            __sync_synchronize();
            QinHost[blockToUse].head ++;
            __sync_synchronize();

            ++nextIns;
        }
        // done, signal all threads to finish and stop 
        for (int block = 0; block < tblocks; ++block)
        {
            while(QinHost[block].tail + QSIZE <= QinHost[block].head) __sync_synchronize();
            QinHost[block].jobs[QinHost[block].head % QSIZE].jobId = DONEJOB;
            __sync_synchronize();
            ++QinHost[block].head;
            __sync_synchronize();
        }
        // printf("syncing\n");
        __sync_synchronize();
        // cudaDeviceSynchronize(); // TODO DO WE NEED THIS
        // printf("done syncing\n");
        // get the rest of the images to cpu
        // printf("So far received: %d jobs\n",amntRecv);
        while(amntRecv < NREQUESTS) {
            cudaDeviceSynchronize();
            __sync_synchronize();
            for (int block = 0; block < tblocks; ++block)
            {
                __sync_synchronize();
                while ( QoutHost[block].tail < QoutHost[block].head )
                {
                    __sync_synchronize();
                    jobS& job = QoutHost[block].jobs[QoutHost[block].tail % QSIZE ];
                    __sync_synchronize();
                    memcpy(images_out_from_gpu + (job.jobId * SQR(IMG_DIMENSION)), job.job, SQR(IMG_DIMENSION) );
                    __sync_synchronize();
                    QoutHost[block].tail++;
                    __sync_synchronize();
                    req_t_end[job.jobId] = get_time_msec();
                    ++amntRecv;
                    // printf("received completed job #%d\n",job.jobId);
                }
            }
        }
        __sync_synchronize();
        printf("In total received: %d jobs\n",amntRecv);
        cudaFreeHost(QinHost);
        cudaFreeHost(QoutHost);
        }
    else {
        assert(0);
    }
    double tf = get_time_msec();

    total_distance = distance_sqr_between_image_arrays(images_out, images_out_from_gpu);
    double avg_latency = 0;
    for (int i = 0; i < NREQUESTS; i++) {
        avg_latency += (req_t_end[i] - req_t_start[i]);
    }
    //printf("Total latency: %f\n",avg_latency); // REMOVE DEBUG
    avg_latency /= NREQUESTS;

    printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", load);
    if (mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", threads_queue_mode);
    printf("distance from baseline %lf (should be zero)\n", total_distance);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);

    return 0;
}
