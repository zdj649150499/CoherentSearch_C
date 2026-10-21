/* Process-wide lock around FFTW *planning* and wisdom I/O.
 *
 * FFTW plan creation mutates global planner state and is not thread-safe; plan
 * *execution* is.  The list driver prepares the next file on a background
 * thread (realfft planning) while the main thread may build search workspaces
 * or candidate-profile plans, so all planning entry points take this lock.
 */
#include "coherent_search.h"
#include <pthread.h>

static pthread_mutex_t cs_fftw_mutex = PTHREAD_MUTEX_INITIALIZER;

void cs_fftw_plan_lock(void)   { pthread_mutex_lock(&cs_fftw_mutex); }
void cs_fftw_plan_unlock(void) { pthread_mutex_unlock(&cs_fftw_mutex); }
