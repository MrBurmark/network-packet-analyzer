
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <omp.h>

#define MAX_SYMS 256
#define TILE 32
#define SIZE 16

//#define PRINT

using namespace std;

// here so things will compile
enum { act_NONE, act_FIRE } ;
struct action_t {
	unsigned int type;
};
struct statetype_t {
      statetype_t	**trans;
      statetype_t*	default_trans;
	  list<action_t*> annotations;
      unsigned int	idx;
};

struct rca_t {
      list<statetype_t*> states;
      statetype_t *start_state;
};


struct cuda_statetype_t {
unsigned short	trans[MAX_SYMS]; // array of indices to other cuda_statetype_t s in a cuda_rca_t.states
};

/* cuda version of a restricted counter automata */
struct cuda_rca_t {
cuda_statetype_t*	states; // pointer valid in cpu only
unsigned char*		annotations;
cuda_statetype_t*	cuda_states; // pointer valid in cuda kernel only
unsigned char*		cuda_annotations;
unsigned int		start_state;
unsigned int		num_states;
};

struct cuda_rules {
	unsigned int capacity;
	unsigned int num_rules;
	cuda_rca_t* rules;
};

struct cuda_packets {
	unsigned int capacity; // amount of memory allocated for packet pointers and lengths
	unsigned int num_packets;
	unsigned char** packets; // pointer to arrays of the packets
	unsigned char** cuda__packets; // pointer to cuda arrays of the packets
	unsigned int* lengths;
};

double time_stamp(void){

	return omp_get_wtime();
}



int cache_attack(int in) {

#ifdef PRINT
	printf("attacking cache\n");
#endif

	int amt = 8*1024*1024; // 32MB (for int) (must be a multiple of 1024)
	int a = 0;

	#pragma omp parallel
	{
		int *I = (int*)malloc(sizeof(int)*amt );
		if (I == NULL) exit(1);
		
		for (int k = 0; k < amt ; ++k)
			I[k] = rand()*in;
	
		for (int k = 0; k < amt ; ++k)
			I[k] = I[I[k]%amt] - in;

		int b = I[I[I[0]%amt]%amt]+2*in;

		free(I);

		#pragma omp critical
		{
			a += b - omp_get_thread_num();
			a = (a+in)%b;
		}
	}
	printf("%d\n", a);
	return a;
}


void split_mod32_median_helper(cuda_packets* packets, unsigned int l0, unsigned int h0) {

	unsigned int l = l0, h = h0;
	unsigned int l2, h2;
	unsigned int mid = (h+l)/64*32;
	unsigned int mid_len;

	if (mid <= l0) return;

	while ( l < mid && h > mid ) {

		mid_len = packets->lengths[mid];

		l2 = l;
		h2 = h;

		while( l2 < h2 ) {

			while ( packets->lengths[l2] < mid_len && l2 < h2 ) 
				++l2;
			while ( packets->lengths[h2] > mid_len && l2 < h2 )
				--h2;

			unsigned int tmp0 = packets->lengths[l2];
			unsigned char* tmp1 = packets->packets[l2];

			packets->lengths[l2] = packets->lengths[h2];
			packets->packets[l2] = packets->packets[h2];

			packets->lengths[h2] = tmp0;
			packets->packets[h2] = tmp1;

			if ( l2 < h2 ) {
				++l2;
				--h2;
			}
		}

		if (h2 > mid)
			h = h2;
		else l = l2;

	}

	split_mod32_median_helper(packets, l0, mid);
	split_mod32_median_helper(packets, mid, h0);
}

void split_mod32_median(cuda_packets* packets) {
	split_mod32_median_helper(packets, 0, packets->num_packets);
}


cuda_packets make_cuda_packets(void) {

	cuda_packets packets = {0, 0, NULL, NULL, NULL};

	return packets;
}

// a very basic append routine for adding strings to packets
int packet_append_len(cuda_packets* packets, const unsigned char* str, unsigned int len) {

	unsigned int str_len = len;  // does not include \n
	unsigned int str_len_pad = str_len%SIZE ? str_len + SIZE - str_len%SIZE : str_len;
	
	//for(str_len = 0; str[str_len] != '\n'; ++str_len);

	if (str_len > 0) {

		// check if need more memory
		if ( packets->num_packets >= packets->capacity) {

			if (packets->capacity > 0) {
				
				unsigned char** new_packets = (unsigned char**)realloc(packets->packets, packets->capacity*2 * sizeof(unsigned char*));
				unsigned char** new_cuda_packets = (unsigned char**)realloc(packets->cuda__packets, packets->capacity*2 * sizeof(unsigned char*));
				unsigned int* new_lengths = (unsigned int*)realloc(packets->lengths, packets->capacity*2 * sizeof(unsigned int));

				if (new_packets == NULL || new_cuda_packets == NULL || new_lengths == NULL) {
					printf("memory reallocation failed\n");
					goto Error;
				}

				packets->packets = new_packets;
				packets->cuda__packets = new_cuda_packets;
				packets->lengths = new_lengths;

				packets->capacity *= 2;

			} else {

				unsigned char** new_packets = (unsigned char**)malloc(16 * sizeof(unsigned char*));
				unsigned char** new_cuda_packets = (unsigned char**)malloc(16 * sizeof(unsigned char*));
				unsigned int* new_lengths = (unsigned int*)malloc(16 * sizeof(unsigned int));

				if (new_packets == NULL || new_cuda_packets == NULL || new_lengths == NULL) {
					printf("memory allocation failed\n");
					goto Error;
				}

				packets->packets = new_packets;
				packets->cuda__packets = new_cuda_packets;
				packets->lengths = new_lengths;

				packets->capacity = 16;
			}
		}

		packets->packets[packets->num_packets] = (unsigned char*)malloc(str_len_pad * sizeof(unsigned char));
		if (packets->packets[packets->num_packets] == NULL) {
			printf("memory allocation failed\n");
			goto Error;
		}
		memcpy(packets->packets[packets->num_packets], str, str_len * sizeof(unsigned char));
		memset(packets->packets[packets->num_packets] + str_len, 0, str_len_pad - str_len);

		packets->lengths[packets->num_packets] = str_len_pad;

		packets->num_packets++;
	}

	return 0;

Error:
	return 1;
}


// basic append routine for adding random strings to packets
int packet_append_rand(cuda_packets* packets, unsigned int max_len) {
	
	unsigned int i;
	unsigned int str_len = rand()%max_len+1;  // does not include \n
	unsigned int str_len_pad = str_len%SIZE ? str_len + SIZE - str_len%SIZE : str_len;
	
	//for(str_len = 0; str[str_len] != '\n'; ++str_len);

	if (str_len > 0) {

		// check if need more memory
		if ( packets->num_packets >= packets->capacity) {

			if (packets->capacity > 0) {
				
				unsigned char** new_packets = (unsigned char**)realloc(packets->packets, packets->capacity*2 * sizeof(unsigned char*));
				unsigned char** new_cuda_packets = (unsigned char**)realloc(packets->cuda__packets, packets->capacity*2 * sizeof(unsigned char*));
				unsigned int* new_lengths = (unsigned int*)realloc(packets->lengths, packets->capacity*2 * sizeof(unsigned int));

				if (new_packets == NULL || new_cuda_packets == NULL || new_lengths == NULL) {
					printf("memory reallocation failed\n");
					goto Error;
				}

				packets->packets = new_packets;
				packets->cuda__packets = new_cuda_packets;
				packets->lengths = new_lengths;

				packets->capacity *= 2;

			} else {

				unsigned char** new_packets = (unsigned char**)malloc(16 * sizeof(unsigned char*));
				unsigned char** new_cuda_packets = (unsigned char**)malloc(16 * sizeof(unsigned char*));
				unsigned int* new_lengths = (unsigned int*)malloc(16 * sizeof(unsigned int));

				if (new_packets == NULL || new_cuda_packets == NULL || new_lengths == NULL) {
					printf("memory allocation failed\n");
					goto Error;
				}

				packets->packets = new_packets;
				packets->cuda__packets = new_cuda_packets;
				packets->lengths = new_lengths;

				packets->capacity = 16;
			}
		}

		packets->packets[packets->num_packets] = (unsigned char*)malloc(str_len_pad * sizeof(unsigned char));
		if (packets->packets[packets->num_packets] == NULL) {
			printf("memory allocation failed\n");
			goto Error;
		}

		for (i = 0; i < str_len; ++i)
			packets->packets[packets->num_packets][i] = rand()%255+1;
		for (i = str_len; i < str_len_pad; ++i)
			packets->packets[packets->num_packets][i] = 0;

		packets->lengths[packets->num_packets] = str_len_pad;

		packets->num_packets++;
	}

	return 0;

Error:
	return 1;
}

int malloc_packets_on_device(cuda_packets* packets) {

	cudaError_t cudaStatus;
	unsigned int i;

	for (i = 0; i < packets->num_packets; ++i) {
		cudaStatus = cudaMalloc((void**)&packets->cuda__packets[i], packets->lengths[i] * sizeof(unsigned char));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return 1;
		}
	}

	return 0;
}

int copy_packets_to_device(cuda_packets* packets) {

	cudaError_t cudaStatus;
	unsigned int i;

	for (i = 0; i < packets->num_packets; ++i) {
		cudaStatus = cudaMemcpy(packets->cuda__packets[i], packets->packets[i], packets->lengths[i] * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}
	}

	return 0;
}

void free_packets_on_device(cuda_packets* packets) {

	unsigned int i;

	for (i = 0; i < packets->num_packets; ++i) {
		cudaFree(packets->cuda__packets[i]);
	}
}


void free_cuda_packets(cuda_packets* packets) {

	for (unsigned int i = 0; i < packets->num_packets; ++i) {

		free(packets->packets[i]);
	}

	if (packets->packets) free(packets->packets);
	if (packets->lengths) free(packets->lengths);

	packets->packets = NULL;
	packets->lengths = NULL;

	packets->capacity = 0;
	packets->num_packets = 0;
}



cuda_rules make_cuda_rules (void) {
	
	cuda_rules rules = {0, 0, NULL};

	return rules;
}

// a very basic append routine for adding a rca_t to cuda_rules
int append_rca_t_to_cuda_rules(cuda_rules* rules, rca_t* rule) {

	cudaError_t cudaStatus;
	cuda_rca_t* cuda_rule;
	unsigned int i;
	list<statetype_t*>::iterator cur_state;

	// check if need more memory
	if ( rules->num_rules >= rules->capacity) {

		if (rules->capacity > 0) {
				
			cuda_rca_t* new_rules = (cuda_rca_t*)realloc(rules->rules, rules->capacity*2 * sizeof(cuda_rca_t));

			if (new_rules == NULL) {
				printf("memory reallocation failed\n");
				goto Error0;
			}

			rules->rules = new_rules;

			rules->capacity *= 2;

		} else {

			cuda_rca_t* new_rules = (cuda_rca_t*)malloc(16 * sizeof(cuda_rca_t));

			if (new_rules == NULL) {
				printf("memory allocation failed\n");
				goto Error0;
			}

			rules->rules = new_rules;

			rules->capacity = 16;
		}
	}

	cuda_rule = &rules->rules[rules->num_rules];
	cuda_rule->states =  NULL;
	cuda_rule->annotations = NULL;
	cuda_rule->cuda_states = NULL;
	cuda_rule->cuda_annotations = NULL;
	cuda_rule->num_states = rule->states.size();

	cuda_rule->start_state = rule->start_state->idx; // / / / / 
	
	cuda_rule->states = (cuda_statetype_t*)malloc(cuda_rule->num_states * sizeof(cuda_statetype_t));
	if (cuda_rule->states == NULL) {
		printf("memory allocation failed\n");
		goto Error1;
	}

	cuda_rule->annotations = (unsigned char*)malloc(cuda_rule->num_states * sizeof(unsigned char));
	if (cuda_rule->annotations == NULL) {
		printf("memory allocation failed\n");
		goto Error1;
	}

	cudaStatus = cudaMalloc((void**)&cuda_rule->cuda_states, cuda_rule->num_states * sizeof(cuda_statetype_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error1;
	}

	cudaStatus = cudaMalloc((void**)&cuda_rule->cuda_annotations, cuda_rule->num_states * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error1;
	}

	for (i = 0, cur_state = rule->states.begin(); cur_state != rule->states.end(); ++cur_state, ++i) { // / / / /

		// set up output for state
		cuda_rule->annotations[i] = act_NONE;

		for (list<action_t*>::iterator l = (*cur_state)->annotations.begin(); l != (*cur_state)->annotations.end(); l++) {
			
			if ((*l)->type == act_FIRE) {
				
				cuda_rule->annotations[i] = act_FIRE;
				break;
			}
		}
		
		// set up transition table
		for (unsigned int k = 0; k < MAX_SYMS; ++k){
			
			if ((*cur_state)->trans[k])
				cuda_rule->states[i].trans[k] = (*cur_state)->trans[k]->idx; // assumes idx in [0, num_states), and list iterator is in-order, and idx is the position of a state in the rule->states
			else
				cuda_rule->states[i].trans[k] = (*cur_state)->default_trans->idx;
		}
	}
	
	// copy rule's states over
	cudaStatus = cudaMemcpy(cuda_rule->cuda_states, cuda_rule->states, cuda_rule->num_states * sizeof(cuda_statetype_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error1;
	}

	// copy rule's annotations over
	cudaStatus = cudaMemcpy(cuda_rule->cuda_annotations, cuda_rule->annotations, cuda_rule->num_states * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error1;
	}

	// could free cuda_rule->states if never going to use it again

	rules->num_rules++;

	return 0;

Error1:
	cudaFree(cuda_rule->cuda_states);
	cudaFree(cuda_rule->cuda_annotations);
	if (cuda_rule->states) free(cuda_rule->states);
	if (cuda_rule->annotations) free(cuda_rule->annotations);

	cuda_rule->cuda_states = NULL;
	cuda_rule->states =  NULL; 
	cuda_rule->start_state = 0;
	cuda_rule->num_states = 0;

Error0:
	return 1;
}

// a very basic append routine for adding a random rule to cuda_rules
int append_rand_rule_to_cuda_rules(cuda_rules* rules, int max_size, int max_trans) {

	cudaError_t cudaStatus;
	cuda_rca_t* cuda_rule;
	int l;

	// check if need more memory
	if ( rules->num_rules >= rules->capacity) {

		if (rules->capacity > 0) {
				
			cuda_rca_t* new_rules = (cuda_rca_t*)realloc(rules->rules, rules->capacity*2 * sizeof(cuda_rca_t));

			if (new_rules == NULL) {
				printf("memory reallocation failed\n");
				goto Error0;
			}

			rules->rules = new_rules;

			rules->capacity *= 2;

		} else {

			cuda_rca_t* new_rules = (cuda_rca_t*)malloc(16 * sizeof(cuda_rca_t));

			if (new_rules == NULL) {
				printf("memory allocation failed\n");
				goto Error0;
			}

			rules->rules = new_rules;

			rules->capacity = 16;
		}
	}

	cuda_rule = &rules->rules[rules->num_rules];
	cuda_rule->states =  NULL;
	cuda_rule->annotations = NULL;
	cuda_rule->cuda_states = NULL;
	cuda_rule->cuda_annotations = NULL;
	cuda_rule->num_states = rand()%max_size + 1;

	cuda_rule->start_state = 0;
	
	cuda_rule->states = (cuda_statetype_t*)malloc(cuda_rule->num_states * sizeof(cuda_statetype_t));
	if (cuda_rule->states == NULL) {
		printf("memory allocation failed\n");
		goto Error1;
	}

	cuda_rule->annotations = (unsigned char*)malloc(cuda_rule->num_states * sizeof(unsigned char));
	if (cuda_rule->annotations == NULL) {
		printf("memory allocation failed\n");
		goto Error1;
	}
	
	cudaStatus = cudaMalloc((void**)&cuda_rule->cuda_states, cuda_rule->num_states * sizeof(cuda_statetype_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error1;
	}

	cudaStatus = cudaMalloc((void**)&cuda_rule->cuda_annotations, cuda_rule->num_states * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error1;
	}

	cuda_rule->start_state = rand()%cuda_rule->num_states;

	for (l = 0; l < cuda_rule->num_states; ++l){
		for (int i = 0; i < MAX_SYMS; ++i)
			cuda_rule->states[l].trans[i] = cuda_rule->start_state;
		for (int i = 0; i < rand()%max_trans; ++i)
			cuda_rule->states[l].trans[rand()%MAX_SYMS] = rand()%cuda_rule->num_states;
		cuda_rule->annotations[l] = rand()%32 ? act_NONE : act_FIRE;
	}


	// copy rule's states over
	cudaStatus = cudaMemcpy(cuda_rule->cuda_states, cuda_rule->states, cuda_rule->num_states * sizeof(cuda_statetype_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error1;
	}

	// copy rule's annotations over
	cudaStatus = cudaMemcpy(cuda_rule->cuda_annotations, cuda_rule->annotations, cuda_rule->num_states * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error1;
	}

	// could free cuda_rule->states if never going to use it again

	rules->num_rules++;

	return 0;

Error1:
	cudaFree(cuda_rule->cuda_states);
	cudaFree(cuda_rule->cuda_annotations);
	if (cuda_rule->states) free(cuda_rule->states);
	if (cuda_rule->annotations) free(cuda_rule->annotations);

	cuda_rule->cuda_states = NULL;
	cuda_rule->states =  NULL; 
	cuda_rule->start_state = 0;
	cuda_rule->num_states = 0;

Error0:
	return 1;
}

void free_cuda_rules(cuda_rules* rules) {

	for (unsigned int i = 0; i < rules->num_rules; ++i) {

		if (rules->rules[i].states) free(rules->rules[i].states);
		if (rules->rules[i].annotations) free(rules->rules[i].annotations);
		cudaFree(rules->rules[i].cuda_states);
		cudaFree(rules->rules[i].cuda_annotations);
	}

	if (rules->rules) free (rules->rules);

	rules->rules = NULL;

	rules->capacity = 0;
	rules->num_rules = 0;
}


__global__ 
void cuda_rca_apply_kernel(cuda_rca_t* rules, unsigned char **packets, const unsigned int *lengths, const unsigned int num_packets, unsigned char* __restrict__ output)
{
	
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int bdim = blockDim.x;
	unsigned char* str, *annotations;
	unsigned int k, l, cur_state, start_state;
	
	cuda_statetype_t* states = (rules + bid)->cuda_states; // a block for each rule, maybe put rules in textures, or shared memory
	start_state = (rules + bid)->start_state;
	annotations = (rules + bid)->cuda_annotations;
	
	for (k = tid; k < num_packets; k += bdim) {
		
		str = packets[k]; // each thread works on some packets
		
		cur_state = start_state; // get pointer to initial state
		
#ifdef PRINT
		printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
			rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
			str[0]);
#endif
		
		if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
			output[k] = 1;
#ifdef PRINT
			printf("\n---Alert---\n\n");
#endif
		}
		
		
		for (l = 0; l < lengths[k]; ++l) // means that the last thing in start must be the total length
		{
			cur_state = states[cur_state].trans[str[l]];

#ifdef PRINT
			printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				m, cur_state->annotation,
				( (str[l] > 0x20 && str[l] < 0x7f) ? str[l] : ' '),
				str[l]);
#endif
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
#ifdef PRINT
				printf("\n---Alert---\n\n");
#endif
			}
		}
	}
}

// addresses the issue of threads accessing memory in a non-coalesced fashion

__global__ 
void cuda_rca_apply_kernel01(cuda_rca_t* rules, unsigned char **packets, const unsigned int *lengths, const unsigned int num_packets, unsigned char* __restrict__ output)
{
	
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int bdim = blockDim.x;
	unsigned char* str, *annotations;
	unsigned int k, l, cur_state, start_state;
	__declspec(align (16)) unsigned char tmp[SIZE];
	
	cuda_statetype_t* states = (rules + bid)->cuda_states; // a block for each rule, maybe put rules in textures, or shared memory
	start_state = (rules + bid)->start_state;
	annotations = (rules + bid)->cuda_annotations;
	
	for (k = tid; k < num_packets; k += bdim) {
		
		str = packets[k]; // each thread works on some packets
		
		cur_state = start_state; // get pointer to initial state
		
#ifdef PRINT
		printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
			rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
			str[0]);
#endif
		
		if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
			output[k] = 1;
#ifdef PRINT
			printf("\n---Alert---\n\n");
#endif
		}

		for (l = 0; l < lengths[k]/SIZE*SIZE; l+=SIZE) 
		{

			// force coalesced memory operations // grab 16 characters at a time
			//asm( "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"( ((unsigned int*)(tmp))[0] ), "=r"( ((unsigned int*)(tmp))[1] ), "=r"( ((unsigned int*)(tmp))[2] ), "=r"( ((unsigned int*)(tmp))[3] ) : "l"( ((unsigned int*)(str+l)) ) );
			asm( "ld.global.v2.u64 {%0, %1}, [%2];" : "=l"( ((long long unsigned int*)(tmp))[0] ), "=l"( ((long long unsigned int*)(tmp))[1] ) : "r"( ((long long unsigned int*)(str+l)) ) );

			//*((long unsigned int*)tmp) = *((long unsigned int*)(str+l)); // only does 8 characters

			cur_state = states[cur_state].trans[tmp[0]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[1]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[2]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[3]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[4]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[5]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[6]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[7]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[8]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[9]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[10]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[11]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[12]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[13]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[14]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[15]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}
			
		}
		
		for (l = lengths[k]/SIZE*SIZE; l < lengths[k]; ++l) 
		{
			
			cur_state = states[cur_state].trans[str[l]];

#ifdef PRINT
			printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				cur_state->trans[str[l]], cur_state->annotation,
				( (str[l] > 0x20 && str[l] < 0x7f) ? str[l] : ' '),
				str[l]);
#endif
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
#ifdef PRINT
				printf("\n---Alert---\n\n");
#endif
			}
		}
	}
}

// addresses the issue of varying string lengths by letting threads grab new strings when they need to, serializes computation when they do
// required padded strings with length a multiple of 16
__global__ 
void cuda_rca_apply_kernel02(cuda_rca_t* rules, unsigned char **packets, const unsigned int *lengths, const unsigned int num_packets, unsigned char* __restrict__ output)
{
	
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int bdim = blockDim.x;
	unsigned char* str, *annotations;
	unsigned int k, l, len, cur_state, start_state;
	__declspec(align (16)) unsigned char tmp[SIZE];
	
	cuda_statetype_t* states = (rules + bid)->cuda_states; // a block for each rule, maybe put rules in textures, or shared memory
	start_state = (rules + bid)->start_state;
	annotations = (rules + bid)->cuda_annotations;

	// initial setup
	k = tid;

	if (k >= num_packets) return; // this thread is done

	str = packets[k]; // each thread works on some packets

	len = lengths[k]; // get the length of the packet
		
	cur_state = start_state; // get pointer to initial state

	l = 0; // prepare index of string
		
#ifdef PRINT
	printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
		rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
		str[0]);
#endif
		
	if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
		output[k] = 1;
#ifdef PRINT
		printf("\n---Alert---\n\n");
#endif
	}


	while (true) {

		if (l >= len) { // move to next packet, serializes, but keeps threads busy if different size packets used

			k += bdim;

			if (k >= num_packets) return; // this thread is done

			str = packets[k]; // thread gets new packet

			len = lengths[k]; // get the length of the packet
		
			cur_state = start_state; // return to initial state

			l = 0; // prepare index of string
		
#ifdef PRINT
			printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
				str[0]);
#endif
		
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
#ifdef PRINT
				printf("\n---Alert---\n\n");
#endif
			}

		}

		{ // continue processing this packet

			// force coalesced memory operations // grab 16 characters at a time
			asm( "ld.global.v2.u64 {%0, %1}, [%2];" : "=l"( ((long long unsigned int*)(tmp))[0] ), "=l"( ((long long unsigned int*)(tmp))[1] ) : "r"( ((long long unsigned int*)(str+l)) ) );

			//*((long unsigned int*)tmp) = *((long unsigned int*)(str+l)); // only does 8 characters

			cur_state = states[cur_state].trans[tmp[0]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[1]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[2]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[3]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[4]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[5]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[6]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[7]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[8]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[9]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[10]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[11]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[12]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[13]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[14]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[15]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}
			
/*
#pragma unroll 16
			for (m = 0; m < SIZE; ++m){ // compiler unrolls this (16 times)

				cur_state = rca->cuda_states + cur_state->trans[tmp[m]];

				//if (tmp[m] != str[l+m]) printf("error, tmp inconsistent with str, %d, %d\n", tmp[m], str[l+m]);

#ifdef PRINT
				printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
					cur_state->trans[tmp[m]], cur_state->annotation,
					( (tmp[m] > 0x20 && tmp[m] < 0x7f) ? tmp[m] : ' '),
					tmp[m]);
#endif
			
				if (cur_state->annotation) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
#ifdef PRINT
					printf("\n---Alert---\n\n");
#endif
				}
			}
*/

			l+=SIZE;

		}
	}
}


// add prefetching
__global__ 
void cuda_rca_apply_kernel03(cuda_rca_t* __restrict__ rules, unsigned char**__restrict__ packets, const unsigned int* __restrict__ lengths, const unsigned int num_packets, unsigned char* __restrict__ output)
{
	
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int bdim = blockDim.x;
	unsigned char* str, *annotations;
	unsigned int k, l, len, cur_state, start_state;
	__declspec(align (16)) unsigned char tmp[SIZE];
	
	cuda_statetype_t* states = (rules + bid)->cuda_states; // a block for each rule, maybe put rules in textures, or shared memory
	start_state = (rules + bid)->start_state;
	annotations = (rules + bid)->cuda_annotations;

	// initial setup
	k = tid;

	if (k >= num_packets) return; // this thread is done

	str = packets[k]; // each thread works on some packets

	len = lengths[k]; // get the length of the packet
		
	cur_state = start_state; // get pointer to initial state

	l = 0; // prepare index of string
		
#ifdef PRINT
	printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
		rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
		str[0]);
#endif
		
	if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
		output[k] = 1;
#ifdef PRINT
		printf("\n---Alert---\n\n");
#endif
	}


	while (true) {

		if (l >= len) { // move to next packet, serializes, but keeps threads busy if different size packets used

			k += bdim;

			if (k >= num_packets) return; // this thread is done

			str = packets[k]; // thread gets new packet

			asm( "prefetch.global.L1 [%0];" : : "r"( str ) );

			len = lengths[k]; // get the length of the packet
		
			cur_state = start_state; // return to initial state

			l = 0; // prepare index of string
		
#ifdef PRINT
			printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
				str[0]);
#endif
		
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
#ifdef PRINT
				printf("\n---Alert---\n\n");
#endif
			}

		}

		{ // continue processing this packet

			// prefetech next
			//asm( "prefetch.global.L2 [%0];" : : "r"( str+l+2*SIZE ) );
			//asm( "prefetch.global.L1 [%0];" : : "r"( str+l+SIZE ) );

			// force coalesced memory operations // grab 16 characters at a time
			asm( "ld.global.v2.u64 {%0, %1}, [%2];" : "=l"( ((long long unsigned int*)(tmp))[0] ), "=l"( ((long long unsigned int*)(tmp))[1] ) : "r"( ((long long unsigned int*)(str+l)) ) );

			//*((long unsigned int*)tmp) = *((long unsigned int*)(str+l)); // only does 8 characters


			cur_state = states[cur_state].trans[tmp[0]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[1]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[2]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[3]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[4]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[5]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[6]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[7]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[8]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[9]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[10]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[11]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[12]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[13]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[14]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state = states[cur_state].trans[tmp[15]];
			
			if (annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			asm( "prefetch.global.L2 [%0];" : : "r"( str+l+SIZE ) );
			
/*
#pragma unroll 16
			for (m = 0; m < SIZE; ++m){ // compiler unrolls this (16 times)

				cur_state = rca->cuda_states + cur_state->trans[tmp[m]];

				//if (tmp[m] != str[l+m]) printf("error, tmp inconsistent with str, %d, %d\n", tmp[m], str[l+m]);

#ifdef PRINT
				printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
					cur_state->trans[tmp[m]], cur_state->annotation,
					( (tmp[m] > 0x20 && tmp[m] < 0x7f) ? tmp[m] : ' '),
					tmp[m]);
#endif
			
				if (cur_state->annotation) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
#ifdef PRINT
					printf("\n---Alert---\n\n");
#endif
				}
			}
*/

			l+=SIZE;

		}
	}
}

/*
// test of thread corsening to use three rules per thread block
// not completed before 2359:59 2012-11-30
// slow

__global__ 
void cuda_rca_apply_kernel04(cuda_rca_t* rules, const unsigned int num_rules, unsigned char **packets, const unsigned int *lengths, const unsigned int num_packets, unsigned char* __restrict__ output)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int bdim = blockDim.x;
	unsigned char *str;
	unsigned int k, l, len;
	__declspec(align (SIZE)) unsigned char tmp[SIZE]; // __attribute__((aligned (16)));
	cuda_statetype_t *cur_state[2], *init_state[2];
	cuda_rca_t* rca[2];

	if (2*bid+1 < num_rules){
		rca[0] = rules + 2*bid + 0;
		rca[1] = rules + 2*bid + 1;
	}
	else if (2*bid+0 < num_rules){
		rca[0] = rules + 2*bid + 0;
		rca[1] = rules + 2*bid + 0;
	}
	else return;


	init_state[0] = rca[0]->cuda_states + rca[0]->start_state;
	init_state[1] = rca[1]->cuda_states + rca[1]->start_state;

	// initial setup
	k = tid;

	if (k >= num_packets) return; // this thread is done

	str = packets[k]; // each thread works on some packets

	len = lengths[k]; // get the length of the packet
	

	cur_state[0] = init_state[0]; // get pointer to initial state
	cur_state[1] = init_state[1]; // get pointer to initial state

	l = 0; // prepare index of string
		
#ifdef PRINT
	printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
		rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
		str[0]);
#endif
	

	if (cur_state[0]->annotation || cur_state[1]->annotation) { // sends the alarm, puts a 1 in that slot of the output
		output[k] = 1;
#ifdef PRINT
		printf("\n---Alert---\n\n");
#endif
	}


	while (true) {

		if (l >= len) { // move to next packet, serializes, but keeps threads busy if different size packets used

			k += bdim;

			if (k >= num_packets) return; // this thread is done

			str = packets[k]; // thread gets new packet

			len = lengths[k]; // get the length of the packet
		
			cur_state[0] = init_state[0]; // return to initial state
			cur_state[1] = init_state[1]; // return to initial state
	
			l = 0; // prepare index of string
		
#ifdef PRINT
			printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
				str[0]);
#endif

			if (cur_state[0]->annotation || cur_state[1]->annotation) { // sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
#ifdef PRINT
				printf("\n---Alert---\n\n");
#endif
			}

		}

		{ // continue processing this packet

			// force coalesced memory operations // grab 16 characters at a time
			asm( "ld.global.v2.u64 {%0, %1}, [%2];" : "=l"( ((long long unsigned int*)(tmp))[0] ), "=l"( ((long long unsigned int*)(tmp))[1] ) : "r"( ((long long unsigned int*)(str+l)) ) );

			//*((long unsigned int*)tmp) = *((long unsigned int*)(str+l)); // only does 8 characters	

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[0]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[0]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[1]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[1]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[2]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[2]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[3]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[3]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[4]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[4]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[5]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[5]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[6]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[6]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[7]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[7]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[8]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[8]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[9]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[9]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[10]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[10]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[11]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[11]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[12]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[12]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[13]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[13]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[14]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[14]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			cur_state[0] = rca[0]->cuda_states + cur_state[0]->trans[tmp[15]];
			cur_state[1] = rca[1]->cuda_states + cur_state[1]->trans[tmp[15]];
			
			if (cur_state[0]->annotation || cur_state[1]->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
			}

			l+=SIZE;

		}
	}
}
*/

// kernels 2, 3 attempted to address memory access problems by having all threads in a warp batch load characters together
// to shared memory before processing, but they were at best as fast as the naive kernel

// compute capability 2.0 required 
/*
__global__ 
void cuda_rca_apply_kernel2(cuda_rca_t* rules, unsigned char **packets, const unsigned int *lengths, const unsigned int num_packets, unsigned char* __restrict__ output)
{
	extern __shared__ unsigned char str_portion[];

	unsigned int tid = threadIdx.x;
	unsigned int tidmod = tid%32;
	unsigned int wst = tid - tidmod; // warp start
	unsigned int bid = blockIdx.x;
	unsigned int bdim = blockDim.x; // divisible by 32
	unsigned int k, l, m, p, r;
	cuda_statetype_t* cur_state;
	
	cuda_rca_t* rca = rules + bid; // a block for each rule
	
	for (k = tid, m = wst; __any(k < num_packets); k += bdim, m += bdim) { // each thread works on some packets

		if (k < num_packets) { // init
		
			cur_state = rca->cuda_states + rca->start_state; // get pointer to initial state

#ifdef PRINT
			printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				rca->start_state, cur_state->annotation, ( (packets[k][0] > 0x20 && packets[k][0] < 0x7f) ? packets[k][0] : ' '),
				packets[k][0]);
#endif
		
			if (cur_state->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
#ifdef PRINT
				printf("\n---Alert---\n\n");
#endif
			}
		}

		for (l = 0; __any(k < num_packets && l < lengths[k]/TILE*TILE); l += TILE) // all threads in a warp enter here until all packets for the warp are done
		{
					
			for (r = 0; m+r < num_packets && r < 32; ++r) { // copy a portion of each packet into shared memory
				
				if (l < lengths[m + r]/TILE*TILE) {

					for (p = tidmod; p < TILE; p+=32)
						str_portion[(wst+r)*TILE+p] = packets[m+r][l+p];
				}
			}

			if (k < num_packets && l < lengths[k]/TILE*TILE) { // only those threads with non-finished packets enter
			
				for (p = tid*TILE; p < tid*TILE + TILE; ++p) {

					r = cur_state->trans[str_portion[p]];

					cur_state = rca->cuda_states + r; 

#ifdef PRINT
					printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
						r, cur_state->annotation,
						( (str_portion[p] > 0x20 && str_portion[p] < 0x7f) ? str_portion[p] : ' '),
						str_portion[p]);
#endif
				
					if (cur_state->annotation) {// sends the alarm, puts a 1 in that slot of the output
						output[k] = 1;
#ifdef PRINT
						printf("\n---Alert---\n\n");
#endif
					}
				}
			}
		}

		for (l = k < num_packets ? lengths[k]/TILE*TILE : 0; __any(k < num_packets && l < lengths[k]); l += TILE) // all threads in a warp enter here for final packet processing stage
		{
			
			for (r = 0; m+r < num_packets && r < 32; ++r) { // copy a portion of each packet into shared memory

				for (p = tidmod; l+p < lengths[m + r] && p < TILE; p+=32)
					str_portion[(wst+r)*TILE+p] = packets[m+r][l+p];
			}

			if (k < num_packets) { // only those threads with packets enter
			
				for (p = tid*TILE; l < lengths[k]; ++p, ++l) { // repeat until packet finished // && p < tid*TILE + TILE

					r = cur_state->trans[str_portion[p]];

					cur_state = rca->cuda_states + r; 

#ifdef PRINT
					printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
						r, cur_state->annotation,
						( (str_portion[p] > 0x20 && str_portion[p] < 0x7f) ? str_portion[p] : ' '),
						str_portion[p]);
#endif
				
					if (cur_state->annotation) {// sends the alarm, puts a 1 in that slot of the output
						output[k] = 1;
#ifdef PRINT
						printf("\n---Alert---\n\n");
#endif
					}
				}
			}
		}
	}
}
*/

// compute capability 3.0 required 
/*
__global__ 
void cuda_rca_apply_kernel3(cuda_rca_t* rules, unsigned char **packets, const unsigned int *lengths, const unsigned int num_packets, unsigned char* __restrict__ output)
{
	extern __shared__ unsigned char str_portion[];

	unsigned int tid = threadIdx.x;
	unsigned int tidmod = tid%32;
	unsigned int wst = tid - tidmod; // warp start
	unsigned int bid = blockIdx.x;
	unsigned int bdim = blockDim.x;
	int k, l, m, p, r;
	cuda_statetype_t* cur_state;
	
	cuda_rca_t* rca = rules + bid; // a block for each rule
	
	for (k = tid, m = wst; __any(k < num_packets); k += bdim, m += bdim) { // each thread works on some packets

		if (k < num_packets) { // init
		
			cur_state = rca->cuda_states + rca->start_state; // get pointer to initial state
		
#ifdef PRINT
			printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				rca->start_state, cur_state->annotation, ( (packets[k][0] > 0x20 && packets[k][0] < 0x7f) ? packets[k][0] : ' '),
				packets[k][0]);
#endif
		
			if (cur_state->annotation) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
#ifdef PRINT
				printf("\n---Alert---\n\n");
#endif
			}
		}

		for (l = 0; __any(k < num_packets && l < lengths[k]/TILE*TILE); l += TILE) // all threads in a warp enter here until all packets for the warp are done
		{
			
			for (r = 0; m+r < num_packets && r < 32; ++r) { // copy a portion of each packet into shared memory
				
				if (__shfl(k < num_packets && l < lengths[k]/TILE*TILE, r)) {

					for (p = tidmod; p < TILE; p+=32)
						str_portion[(wst+r)*TILE+p] = packets[m+r][l+p]; //m+r
				}
			}

			if (k < num_packets && l < lengths[k]/TILE*TILE) { // only those threads with non-finished packets enter
			
				for (p = tid*TILE; p < tid*TILE + TILE; ++p) {

					r = cur_state->trans[str_portion[p]];

					cur_state = rca->cuda_states + r; 

#ifdef PRINT
					printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
						r, cur_state->annotation,
						( (str_portion[p] > 0x20 && str_portion[p] < 0x7f) ? str_portion[p] : ' '),
						str_portion[p]);
#endif
				
					if (cur_state->annotation) {// sends the alarm, puts a 1 in that slot of the output
						output[k] = 1;
#ifdef PRINT
						printf("\n---Alert---\n\n");
#endif
					}
				}
			}
		}

		for (l = k < num_packets ? lengths[k]/TILE*TILE : 0; __any(k < num_packets && l < lengths[k]); l += TILE) // all threads in a warp enter here for final packet processing stage
		{
			
			for (r = 0; m+r < num_packets && r < 32; ++r) { // copy a portion of each packet into shared memory

				if (__shfl(k < num_packets && l < lengths[k], r)) {

					for (p = tidmod; p < TILE; p+=32)
						str_portion[(wst+r)*TILE+p] = packets[m+r][l+p]; // m+r
				}
			}

			if (k < num_packets) { // only those threads with packets enter
			
				for (p = tid*TILE; l < lengths[k]; ++p, ++l) { // repeat until packet finished // && p < tid*TILE + TILE

					r = cur_state->trans[str_portion[p]];

					cur_state = rca->cuda_states + r; 

#ifdef PRINT
					printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
						r, cur_state->annotation,
						( (str_portion[p] > 0x20 && str_portion[p] < 0x7f) ? str_portion[p] : ' '),
						str_portion[p]);
#endif
				
					if (cur_state->annotation) {// sends the alarm, puts a 1 in that slot of the output
						output[k] = 1;
#ifdef PRINT
						printf("\n---Alert---\n\n");
#endif
					}
				}
			}
		}
	}
}
*/



cudaError_t cuda_dfa(const cuda_rules* rules, const cuda_packets* packets, unsigned char* output, const int ker)
{
	unsigned int i;
	cuda_rca_t* dev_rules;
	unsigned char** dev_packets;
	unsigned int* dev_lengths;
	unsigned char* dev_output;
	unsigned int num_threads = packets->num_packets > 1024 ? 1024 : packets->num_packets;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_rules, rules->num_rules * sizeof(cuda_rca_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	for (i = 0; i < packets->num_packets; ++i) {
		cudaStatus = cudaMalloc((void**)&packets->cuda__packets[i], packets->lengths[i] * sizeof(unsigned char));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}

	cudaStatus = cudaMalloc((void**)&dev_packets, packets->num_packets * sizeof(unsigned char*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_lengths, packets->num_packets * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output, packets->num_packets * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_rules, rules->rules, rules->num_rules * sizeof(cuda_rca_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	for (i = 0; i < packets->num_packets; ++i) {
		cudaStatus = cudaMemcpy(packets->cuda__packets[i], packets->packets[i], packets->lengths[i] * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}

	cudaStatus = cudaMemcpy(dev_packets, packets->cuda__packets, packets->num_packets * sizeof(unsigned char*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_lengths, packets->lengths, packets->num_packets * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_output, output, packets->num_packets * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	switch(ker) {
	case(0) :
		cuda_rca_apply_kernel<<<rules->num_rules, num_threads>>>(dev_rules, dev_packets, dev_lengths, packets->num_packets, dev_output);
		break;
	case(1) :
		cuda_rca_apply_kernel01<<<rules->num_rules, num_threads>>>(dev_rules, dev_packets, dev_lengths, packets->num_packets, dev_output);
		break;
	case(2) :
		cuda_rca_apply_kernel02<<<rules->num_rules, num_threads>>>(dev_rules, dev_packets, dev_lengths, packets->num_packets, dev_output);
		break;
	case(3) :
		cuda_rca_apply_kernel02<<<rules->num_rules, num_threads>>>(dev_rules, dev_packets, dev_lengths, packets->num_packets, dev_output);
		break;
	case(4) : 
		//cuda_rca_apply_kernel04<<<rules->num_rules%2 ? rules->num_rules/2 + 1 : rules->num_rules/2, num_threads>>>(dev_rules, rules->num_rules, dev_packets, dev_lengths, packets->num_packets, dev_output);
		break;
	case(5) :
		//cuda_rca_apply_kernel2<<<rules->num_rules, (num_threads+31)/32*32, TILE*num_threads*sizeof(unsigned char)>>>(dev_rules, dev_packets, dev_lengths, packets->num_packets, dev_output);
		break;
	case(6) :
		//cuda_rca_apply_kernel3<<<rules->num_rules, (num_threads+31)/32*32, TILE*num_threads*sizeof(unsigned char)>>>(dev_rules, dev_packets, dev_lengths, packets->num_packets, dev_output);
		break;
	}


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output, dev_output, packets->num_packets * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	/*
	for (i = 0; i < packets->num_packets; ++i) {
		cudaFree(packets->cuda__packets[i]);
	}
	*/
	cudaFree(dev_rules);
	cudaFree(dev_packets);
	cudaFree(dev_lengths);
	cudaFree(dev_output);

	return cudaStatus;
}


int serial_dfa(const cuda_rules* rules, const cuda_packets* packets, unsigned char* output)
{
	for (unsigned int i = 0; i < rules->num_rules; ++i) {

		cuda_rca_t* rca = rules->rules + i;

		for (unsigned int k = 0; k < packets->num_packets; ++k) {

			unsigned char* str = packets->packets[k];

			unsigned int cur_state = rca->start_state; // get pointer to initial state

#ifdef PRINT
			printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
				str[0]);
#endif

			if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
#ifdef PRINT
				printf("\n---Alert---\n\n");
#endif
			}

			for (unsigned int l = 0; l < packets->lengths[k]; ++l) // means that the last thing in start must be the total length
			{

				cur_state = rca->states[cur_state].trans[str[l]]; 

#ifdef PRINT
				printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
					cur_state - rca->start_state, cur_state->annotation,
					( (str[i] > 0x20 && str[i] < 0x7f) ? str[i] : ' '),
					str[i]);
#endif

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
#ifdef PRINT
					printf("\n---Alert---\n\n");
#endif
				}
			}
		}
	}
	return 0;
}


#define R 8
#define P 8

void serial_dfa1(const cuda_rules* rules, const cuda_packets* packets, unsigned char* output)
{

	for (unsigned int r = 0; r < rules->num_rules; r+=R)
	for (unsigned int p = 0; p < packets->num_packets; p+=P)
	for (unsigned int i = r; i < r+R && i < rules->num_rules; ++i) {

		cuda_rca_t* rca = rules->rules + i;

		for (unsigned int k = p; k < p+P && k < packets->num_packets; ++k) {

			unsigned char* str = packets->packets[k];
		
			unsigned int cur_state = rca->start_state; // get pointer to initial state
		
#ifdef PRINT
			printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
				str[0]);
#endif
		
			if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
#ifdef PRINT
				printf("\n---Alert---\n\n");
#endif
			}
			
			for (unsigned int l = 0; l < packets->lengths[k]; ++l) // means that the last thing in start must be the total length
			{
			
				cur_state = rca->states[cur_state].trans[str[l]]; 

#ifdef PRINT
				printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
					cur_state->trans[str[l]], cur_state->annotation,
					( (str[l] > 0x20 && str[l] < 0x7f) ? str[l] : ' '),
					str[l]);
#endif
			
				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
#ifdef PRINT
					//printf("\n---Alert---\n\n");
#endif
				}
			}
		}
	}
}

#define AMT 16

void serial_dfa2(const cuda_rules* rules, const cuda_packets* packets, unsigned char* output)
{

	for (unsigned int r = 0; r < rules->num_rules; r+=R)
	for (unsigned int p = 0; p < packets->num_packets; p+=P)
	for (unsigned int i = r; i < r+R && i < rules->num_rules; ++i) {

		cuda_rca_t* rca = rules->rules + i;

		for (unsigned int k = p; k < p+P && k < packets->num_packets; ++k) {

			unsigned char* str = packets->packets[k];
		
			unsigned int cur_state = rca->start_state; // get pointer to initial state
		
#ifdef PRINT
			printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
				str[0]);
#endif
		
			if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
				//printf("\n---Alert---\n\n");
			}

			for (unsigned int l = 0; l < packets->lengths[k]/AMT*AMT; l+=16) 
			{
			
				cur_state = rca->states[cur_state].trans[str[l+0]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+1]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+2]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+3]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+4]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+5]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+6]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+7]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+8]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+9]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+10]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+11]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+12]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+13]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+14]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+15]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}
			}

			for (unsigned int l = packets->lengths[k]/AMT*AMT; l < packets->lengths[k]; ++l)
			{
			
				cur_state = rca->states[cur_state].trans[str[l]]; 

#ifdef PRINT
				printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
					cur_state->trans[str[l]], cur_state->annotation,
					( (str[l] > 0x20 && str[l] < 0x7f) ? str[l] : ' '),
					str[l]);
#endif
			
				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
#ifdef PRINT
					printf("\n---Alert---\n\n");
#endif
				}
			}
		}
	}
}


void parallel_dfa(const cuda_rules* rules, const cuda_packets* packets, unsigned char* output, int num_threads)
{
#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < rules->num_rules; ++i) {

		cuda_rca_t* rca = rules->rules + i;

		for (unsigned int k = 0; k < packets->num_packets; ++k) {

			unsigned char* str = packets->packets[k];

			unsigned int cur_state = rca->start_state; // get pointer to initial state

#ifdef PRINT
			printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
				str[0]);
#endif

			if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
#ifdef PRINT
				printf("\n---Alert---\n\n");
#endif
			}

			for (unsigned int l = 0; l < packets->lengths[k]; ++l) // means that the last thing in start must be the total length
			{

				cur_state = rca->states[cur_state].trans[str[l]]; 

#ifdef PRINT
				printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
					cur_state - rca->start_state, cur_state->annotation,
					( (str[i] > 0x20 && str[i] < 0x7f) ? str[i] : ' '),
					str[i]);
#endif

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
#ifdef PRINT
					printf("\n---Alert---\n\n");
#endif
				}
			}
		}
	}
}

void parallel_dfa1(const cuda_rules* rules, const cuda_packets* packets, unsigned char* output, int num_threads)
{

#pragma omp parallel for num_threads(num_threads)
	for (int r = 0; r < rules->num_rules; r+=R)
	for (unsigned int p = 0; p < packets->num_packets; p+=P)
	for (unsigned int i = r; i < r+R && i < rules->num_rules; ++i) {

		cuda_rca_t* rca = rules->rules + i;

		for (unsigned int k = p; k < p+P && k < packets->num_packets; ++k) {

			unsigned char* str = packets->packets[k];
		
			unsigned int cur_state = rca->start_state; // get pointer to initial state
		
#ifdef PRINT
			printf("Set initial state: %d: (%u actions) on symbol: '%c' (%d)\n", 
				rca->start_state, cur_state->annotation, ( (str[0] > 0x20 && str[0] < 0x7f) ? str[0] : ' '),
				str[0]);
#endif
		
			if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
				output[k] = 1;
#ifdef PRINT
				printf("\n---Alert---\n\n");
#endif
			}

			for (unsigned int l = 0; l < packets->lengths[k]/AMT*AMT; l+=16) 
			{
			
				cur_state = rca->states[cur_state].trans[str[l+0]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+1]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+2]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+3]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+4]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+5]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+6]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+7]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+8]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+9]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+10]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+11]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+12]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+13]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+14]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}

				cur_state = rca->states[cur_state].trans[str[l+15]]; 

				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
				}
			}

			for (unsigned int l = packets->lengths[k]/AMT*AMT; l < packets->lengths[k]; ++l)
			{
			
				cur_state = rca->states[cur_state].trans[str[l]]; 

#ifdef PRINT
				printf("Moved to state: %d: (%u actions) on symbol: '%c' (%d)\n", 
					cur_state->trans[str[l]], cur_state->annotation,
					( (str[l] > 0x20 && str[l] < 0x7f) ? str[l] : ' '),
					str[l]);
#endif
			
				if (rca->annotations[cur_state] != act_NONE) {// sends the alarm, puts a 1 in that slot of the output
					output[k] = 1;
#ifdef PRINT
					printf("\n---Alert---\n\n");
#endif
				}
			}
		}
	}
}


int main(int argc, char** argv)
{
	cudaError_t cudaStatus;
	unsigned int err = 0;
	cuda_packets packets;
	cuda_rules rules;
	unsigned char* cuda_output = 0;
	unsigned char* serial_output = 0;
	unsigned char* parallel_output = 0;
	double t_cuda_mal, t_cuda_cpy, t_cuda_free, t_cuda0, t_cuda1, t_cuda2, t_cuda3, t_cuda4, t_cuda5, t_cuda6, 
		t_serial0, t_serial1, t_serial2, 
		t_parallel1, t_parallel2, t_parallel4, t_parallel8, t_parallel11, t_parallel12, t_parallel14, t_parallel18, 
		t_rule, t_packet, t_sort, t_tmp;
	unsigned int i, k;
	unsigned int packet_bytes = 0;


	unsigned int num_rules = 678;
	unsigned int num_packets = 1735;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	cudaStatus = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSetCacheConfig failed!");
		return 1;
	}

	t_rule = time_stamp();
	srand( (unsigned int)( (t_rule - floorf(t_rule)) * 1e6f ) );


	printf("appending rules\n");

	t_rule = time_stamp();

	rules = make_cuda_rules();

	for (i = 0; i < num_rules; ++i) {
		err += append_rand_rule_to_cuda_rules(&rules, 278, rand()%16+1); // a larger rule as we mostly have small rules
	}

	t_rule = time_stamp() - t_rule;


	printf("appending packets\n");	

	t_packet = time_stamp();
	
	packets = make_cuda_packets();

	for (i = 0; i < num_packets; ++i) { 
		err += packet_append_rand(&packets, 1500);
		packet_bytes += packets.lengths[packets.num_packets-1] * sizeof(unsigned char);
	}

	if (err)  {
		fprintf(stderr, "malloc failed!");
		return 1;
	}

	t_packet = time_stamp() - t_packet;




	printf("starting serial test\n");

	serial_output = (unsigned char*)malloc(packets.num_packets * sizeof(unsigned char));
	if (serial_output == NULL) {
		printf("memory allocation failed\n");
		return 1;
	}

	cache_attack(47);

	for (i = 0; i < packets.num_packets; ++i)
		serial_output[i] = 0;

	t_serial0 = time_stamp();
	serial_dfa(&rules, &packets, serial_output);
	t_serial0 = time_stamp() - t_serial0;

	cache_attack(54);

	for (i = 0; i < packets.num_packets; ++i)
		serial_output[i] = 0;

	t_serial1 = time_stamp();
	serial_dfa1(&rules, &packets, serial_output);
	t_serial1 = time_stamp() - t_serial1;

	cache_attack(7);

	for (i = 0; i < packets.num_packets; ++i)
		serial_output[i] = 0;

	t_serial2 = time_stamp();
	serial_dfa2(&rules, &packets, serial_output);
	t_serial2 = time_stamp() - t_serial2;





	printf("starting parallel test\n");

	parallel_output = (unsigned char*)malloc(packets.num_packets * sizeof(unsigned char));
	if (parallel_output == NULL) {
		printf("memory allocation failed\n");
		return 1;
	}


	cache_attack(2);

	for (i = 0; i < packets.num_packets; ++i)
		parallel_output[i] = 0;

	t_parallel1 = time_stamp();
	parallel_dfa(&rules, &packets, parallel_output, 1);
	t_parallel1 = time_stamp() - t_parallel1;

	for (i = 0; i < packets.num_packets; ++i)
		parallel_output[i] = 0;

	cache_attack(4);

	t_parallel2 = time_stamp();
	parallel_dfa(&rules, &packets, parallel_output, 2);
	t_parallel2 = time_stamp() - t_parallel2;

	for (i = 0; i < packets.num_packets; ++i)
		parallel_output[i] = 0;

	cache_attack(5);

	t_parallel4 = time_stamp();
	parallel_dfa(&rules, &packets, parallel_output, 4);
	t_parallel4 = time_stamp() - t_parallel4;

	for (i = 0; i < packets.num_packets; ++i)
		parallel_output[i] = 0;

	cache_attack(16);

	t_parallel8 = time_stamp();
	parallel_dfa(&rules, &packets, parallel_output, 8);
	t_parallel8 = time_stamp() - t_parallel8;



	cache_attack(1);

	for (i = 0; i < packets.num_packets; ++i)
		parallel_output[i] = 0;

	t_parallel11 = time_stamp();
	parallel_dfa1(&rules, &packets, parallel_output, 1);
	t_parallel11 = time_stamp() - t_parallel11;

	for (i = 0; i < packets.num_packets; ++i)
		parallel_output[i] = 0;

	cache_attack(10);

	t_parallel12 = time_stamp();
	parallel_dfa1(&rules, &packets, parallel_output, 2);
	t_parallel12 = time_stamp() - t_parallel12;

	for (i = 0; i < packets.num_packets; ++i)
		parallel_output[i] = 0;

	cache_attack(13);

	t_parallel14 = time_stamp();
	parallel_dfa1(&rules, &packets, parallel_output, 4);
	t_parallel14 = time_stamp() - t_parallel14;

	for (i = 0; i < packets.num_packets; ++i)
		parallel_output[i] = 0;

	cache_attack(9);

	t_parallel18 = time_stamp();
	parallel_dfa1(&rules, &packets, parallel_output, 8);
	t_parallel18 = time_stamp() - t_parallel18;




	printf("sorting packets\n");

	t_sort = time_stamp();

	split_mod32_median(&packets);

	t_sort = time_stamp() - t_sort;




	cuda_output = (unsigned char*)malloc(packets.num_packets * sizeof(unsigned char));
	if (cuda_output == NULL) {
		printf("memory allocation failed\n");
		return 1;
	}

	t_cuda_mal = time_stamp();
	if (malloc_packets_on_device(&packets))
		return 1;
	t_cuda_mal = time_stamp() - t_cuda_mal;

	t_cuda_cpy = time_stamp();
	if (copy_packets_to_device(&packets))
		return 1;
	t_cuda_cpy = time_stamp() - t_cuda_cpy;



	printf("starting cuda test 0\n");

	t_cuda0 = 0.0;
	for (k = 0; k < 16; ++k) {

		cache_attack(3 + k*i);

		for (i = 0; i < packets.num_packets; ++i)
			cuda_output[i] = 0;

		t_tmp = time_stamp();
		cudaStatus = cuda_dfa(&rules, &packets, cuda_output, 0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cuda DFA failed!\n");
			return 1;
		}
		t_tmp = time_stamp() - t_tmp;

		t_cuda0 += t_tmp;
	}
	t_cuda0 /= 16.0;


	printf("starting cuda test 1\n");

	t_cuda1 = 0.0;
	for (k = 0; k < 16; ++k) {

		for (i = 0; i < packets.num_packets; ++i)
			cuda_output[i] = 0;

		cache_attack(78 + k*i);

		t_tmp = time_stamp();
		cudaStatus = cuda_dfa(&rules, &packets, cuda_output, 1);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cuda DFA failed!\n");
			return 1;
		}
		t_tmp = time_stamp() - t_tmp;

		t_cuda1 += t_tmp;
	}
	t_cuda1 /= 16.0;


	printf("starting cuda test 2\n");

	t_cuda2 = 0.0;
	for (k = 0; k < 16; ++k) {

		cache_attack(997 + k*i);

		for (i = 0; i < packets.num_packets; ++i)
			cuda_output[i] = 0;

		t_tmp = time_stamp();
		cudaStatus = cuda_dfa(&rules, &packets, cuda_output, 2);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cuda DFA failed!\n");
			return 1;
		}
		t_tmp = time_stamp() - t_tmp;

		t_cuda2 += t_tmp;
	}
	t_cuda2 /= 16.0;

	printf("starting cudatest 3\n");

	t_cuda3 = 0.0;
	for (k = 0; k < 16; ++k) {

		cache_attack(41 + i*k);

		for (i = 0; i < packets.num_packets; ++i)
			cuda_output[i] = 0;

		t_tmp = time_stamp();
		cudaStatus = cuda_dfa(&rules, &packets, cuda_output, 3); // not completed by deadline
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cuda DFA failed!\n");
			return 1;
		}
		t_tmp = time_stamp() - t_tmp;

		t_cuda3 += t_tmp;
	}
	t_cuda3 /= 16.0;

	/*
	printf("starting cuda test 4\n");

	t_cuda4 = 0.0;
	for (k = 0; k < 16; ++k) {

		cache_attack(10 + i*k);

		for (i = 0; i < packets.num_packets; ++i)
			cuda_output[i] = 0;

		t_tmp = time_stamp();
		cudaStatus = cuda_dfa(&rules, &packets, cuda_output, 4); // not completed by deadline
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cuda DFA failed!\n");
			return 1;
		}
		t_tmp = time_stamp() - t_tmp;

		t_cuda4 += t_tmp;
	}
	t_cuda4 /= 16.0;

	printf("starting cuda test 5\n");

	t_cuda5 = 0.0;
	for (k = 0; k < 16; ++k) {

		cache_attack(49 + i*k);

		for (i = 0; i < packets.num_packets; ++i)
			cuda_output[i] = 0;

		t_tmp = time_stamp();
		cudaStatus = cuda_dfa(&rules, &packets, cuda_output, 5); // not completed by deadline
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cuda DFA failed!\n");
			return 1;
		}
		t_tmp = time_stamp() - t_tmp;

		t_cuda5 += t_tmp;
	}
	t_cuda5 /= 16.0;

	printf("starting cuda test 6\n");

	t_cuda6 = 0.0;
	for (k = 0; k < 16; ++k) {

		cache_attack(41 + i*k);

		for (i = 0; i < packets.num_packets; ++i)
			cuda_output[i] = 0;

		t_tmp = time_stamp();
		cudaStatus = cuda_dfa(&rules, &packets, cuda_output, 6); // not completed by deadline
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cuda DFA failed!\n");
			return 1;
		}
		t_tmp = time_stamp() - t_tmp;

		t_cuda6 += t_tmp;
	}
	t_cuda6 /= 16.0;
	*/

	t_cuda_free = time_stamp();
	free_packets_on_device(&packets);
	t_cuda_free = time_stamp() - t_cuda_free;




	for (i = 0; i < packets.num_packets; ++i) {
		
		if (cuda_output[i] != serial_output[i]) {
			printf("packet %d - %s\n", i, "--ERROR-- cuda and serial disagree");
		}
		if (parallel_output[i] != serial_output[i]) {
			printf("packet %d - %s\n", i, "--ERROR-- parallel and serial disagree");
		}
		//printf("packet %d - %s\n", i, serial_output[i] ? "-=#=-MATCH-=#=-" : "  --miss--  ");
	}

	/*
	The versions used on the slides are serial0, parallel 2, parallel 4, cuda1, and cuda2 (optimized, sort).
	*/

	printf("num rules %d, rule append time %f\n", rules.num_rules, t_rule); // rule compilation is considered an offline process so we do not include this in the total time
	printf("num packets %d, total size %dbytes, packet append time %f\n", packets.num_packets, packet_bytes, t_packet); // only has time for malloc and memcpy
	printf("serial0 DFA time %f, %f\n", t_serial0, t_serial0+t_packet);
	printf("serial1 DFA time %f, %f\n", t_serial1, t_serial1+t_packet);
	printf("serial2 DFA time %f, %f\n", t_serial2, t_serial2+t_packet);
	printf("parallel 1 DFA time %f, %f\n", t_parallel1, t_parallel1+t_packet);
	printf("parallel 2 DFA time %f, %f\n", t_parallel2, t_parallel2+t_packet);
	printf("parallel 4 DFA time %f, %f\n", t_parallel4, t_parallel4+t_packet);
	printf("parallel 8 DFA time %f, %f\n", t_parallel8, t_parallel8+t_packet);
	printf("parallel1 1 DFA time %f, %f\n", t_parallel11, t_parallel11+t_packet);
	printf("parallel1 2 DFA time %f, %f\n", t_parallel12, t_parallel12+t_packet);
	printf("parallel1 4 DFA time %f, %f\n", t_parallel14, t_parallel14+t_packet);
	printf("parallel1 8 DFA time %f, %f\n", t_parallel18, t_parallel18+t_packet);
	printf("sort time %f\n", t_sort);
	printf("cuda malloc time %f, copy time %f, free time %f\n", t_cuda_mal, t_cuda_cpy, t_cuda_free);
	printf("cuda0 DFA time %f, %f\n", t_cuda0+t_packet+t_sort+t_cuda_cpy, t_cuda0+t_packet+t_sort+t_cuda_mal+t_cuda_cpy+t_cuda_free); // includes cudamalloc and cudamemcpy for packets
	printf("cuda1 DFA time %f, %f\n", t_cuda1+t_packet+t_sort+t_cuda_cpy, t_cuda1+t_packet+t_sort+t_cuda_mal+t_cuda_cpy+t_cuda_free); // includes cudamalloc and cudamemcpy for packets
	printf("cuda2 DFA time %f, %f\n", t_cuda2+t_packet+t_sort+t_cuda_cpy, t_cuda2+t_packet+t_sort+t_cuda_mal+t_cuda_cpy+t_cuda_free); // includes cudamalloc and cudamemcpy for packets
	printf("cuda3 DFA time %f, %f\n", t_cuda3+t_packet+t_sort+t_cuda_cpy, t_cuda3+t_packet+t_sort+t_cuda_mal+t_cuda_cpy+t_cuda_free); // includes cudamalloc and cudamemcpy for packets // not completed by deadline
	//printf("cuda4 DFA time %f, %f\n", t_cuda4+t_packet+t_sort+t_cuda_cpy, t_cuda4+t_packet+t_sort+t_cuda_mal+t_cuda_cpy+t_cuda_free); // includes cudamalloc and cudamemcpy for packets // not completed by deadline
	//printf("cuda5 DFA time %f, %f\n", t_cuda5+t_packet+t_sort+t_cuda_cpy, t_cuda5+t_packet+t_sort+t_cuda_mal+t_cuda_cpy+t_cuda_free); // includes cudamalloc and cudamemcpy for packets
	//printf("cuda6 DFA time %f, %f\n", t_cuda6+t_packet+t_sort+t_cuda_cpy, t_cuda6+t_packet+t_sort+t_cuda_mal+t_cuda_cpy+t_cuda_free); // includes cudamalloc and cudamemcpy for packets


	if (cuda_output) free(cuda_output);
	if (serial_output) free(serial_output);
	if (parallel_output) free(parallel_output);
	free_cuda_rules(&rules);
	free_cuda_packets(&packets);


	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}



	int a;
	scanf("%d", &a);



	return 0;
}

