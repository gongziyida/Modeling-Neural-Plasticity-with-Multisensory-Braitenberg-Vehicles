#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#define PI 3.141592654
#define TWO_PI 3.141592654 * 2
#define RMNAN(x) if (isnan(*x)) *x = 0 

// General-usage functions begin

/*
 * Gaussian random value generator using Box–Muller transform
 */
double gaussian(double m, double std){
	static double U, V;
	static int phase = 0;

	double z;

	if (phase == 0){
		// random presentages
		U = rand() / (RAND_MAX + 1.0); 
		V = rand() / (RAND_MAX + 1.0);
		
		// z score
		z = sqrt(-2.0 * log(U))* sin(2.0 * PI * V);

	} else{
		z = sqrt(-2.0 * log(U)) * cos(2.0 * PI * V);
	}

	phase = 1 - phase;
	z = z * std + m;
	return z;
}


/*
 * calculate the outer product, or the triangular part of 
 * the outer product
 */
void sq_outer(double *res, const double *a1, const double *a2, 
				int len, char tri){
	int k;

	// upper triangular part
	if (tri == 'u' || tri == 'n'){
		for (int j = 0; j < len; j++){
			for (int i = j; i < len; i++){
				k = i * len + j;
				res[k] = a1[j] * a2[i];

				RMNAN(&res[k]);
				// if (isnan(res[k])){
				// 	res[k] = 0;
				// }
			}
		}
	} 

	// lower triangular part
	if (tri == 'l' || tri == 'n'){
		for (int i = 0; i < len; i++){
			for (int j = i; j < len; j++){
				k = i * len + j;
				res[k] = a1[j] * a2[i];

				RMNAN(&res[k]);
				// if (isnan(res[k])){
				// 	res[k] = 0;
				// }
			}
		}
	}
}


/*
 * calculate I - a * L, where I is an identity matrix and a is a scalar
 */
void I_minus_aL(double *L, double a, int size){
	int k;
	double r;
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			k = i + j * size;
			r = a * L[k];

			// diagonal
			if (j == i){
				L[k] = 1 - r;
			
			} else{ // off-diagonal
				L[k] = - r;
			}

			RMNAN(&L[k]);
			// if (isnan(L[k])){
			// 	L[k] = 0;
			// }
		}
	}
}


/*
 * remove NaN in a matrix
 */
void rmnan(double *w, int *shape){
	int size_in = shape[0];
	int size_out = shape[1];
	int k;

	for (int i = 0; i < size_out; i++){
		for (int j = 0; j < size_in; j++){
			k = i + j * size_out;
			
			RMNAN(&w[k]);
			// if (isnan(w[k])){
			// 	w[k] = 0;
			// }
		}
	}
}

// General-usage functions end

//*****************************************************************

// Functions specified for certain classes begin

/*
 * Select positions of the stimulus sources for class Space.Space 
 */
void set_stim_pos(int *pos, int num_stim, int max_pos, char method){
	if (method == 'r'){ // random mode: set rand seed
		srand((unsigned int) time(NULL));
	} 

	int r = (int) sqrt(num_stim);	// row
	int c = num_stim / r;			// column
	int rmd = num_stim - r * c;		// remainders: the number of pos in the last row

	// if there are remainders, row + 1
    if (rmd != 0){
        int t = rmd / c;
        r += t + 1;
        rmd -= t * c;
    }

    // calculate the intervals between sources, in x and y dimensions
    int xitvl = (int) ceil((double) max_pos / c);
    int yitvl = (int) ceil((double) max_pos / r);
    int x = 0;
    int y = 0;

    // generate the positions
    int i = 0;
    for (int ri = 0; ri < r; y += yitvl, x = 0){
    	for (int ci = 0; ci < c; ci++, x += xitvl, i += 2){
			if (i >= num_stim * 2) return;
			pos[i] = x;
			pos[i+1] = y;
			
			// make random offset
			if (method == 'r'){
				pos[i] += rand() % xitvl;
				pos[i+1] += rand() % yitvl;
			}
    	}
   	}
}


/*
 * Build the space for class Space.Space 
 */

// helper function
double diff(int d, double max_pos){
	d = abs(d);
	if (d > max_pos / 2) {
		return max_pos - d;
	}
	return d;
}

void build_space(double *space, const double *att, const int *pos, 
				int num_stim, int max_pos, int num_orn, 
				int pxl_dim, int gus_T) {

	double xdiff, ydiff, dist, factor;
	double max_pos_d = (double) max_pos;
	int si, ai;

	for (int i = 0; i < max_pos; i++){
		for (int j = 0; j < max_pos; j++){
			for (int k = 0; k < num_stim; k++){
				// for each pixel and each stimulus pos, calculate the distance
				xdiff = diff(pos[k * 2] - i, max_pos_d);
				ydiff = diff(pos[k * 2 + 1] - j, max_pos_d);
				dist = pow(xdiff, 2);
				dist += pow(ydiff, 2);
				dist = sqrt(dist);

				// decaying factor
				factor = exp(-dist / max_pos_d * 10);

				// assign olfactory values
				for (int a = 0; a < num_orn; a++){
					si = (i * max_pos + j) * pxl_dim + a;
					ai = k * pxl_dim + a;
					space[si] += factor * att[ai];

					// value too small, so ignore it
					if (k == num_stim - 1 && space[si] < 1e-6){
						space[si] = 0;
					}
				}

				// assign gustatory values
				if (dist < gus_T){
					for (int a = num_orn; a < pxl_dim; a++){
						si = (i * max_pos + j) * pxl_dim + a;
						ai = k * pxl_dim + a;
						space[si] += att[ai];
					}
				}
			}
		}
	}
}


/*
 * the tanh function for Layers.LiHopfield
 */
void tanh_f(double *res, const double *arr, int len, int ax, double th){
	double s, s_, r, factor;
	
	// constants as described by Li & Hopfield on their original paper
	if (ax == 0){
		s_ = 0.14;
	} else if (ax == 1){
		s_ = 0.29;
	}

	// calculate
	for (int i = 0; i < len; i++){
		r = arr[i] - th;

		if (r < 0){
			s = s_;
		} else{
			s = s_ * 10;
		}

		res[i] = s_ + s * tanh(r / s);

		RMNAN(&res[i]);
		// if (isnan(res[i])){
		// 	res[i] = 0;
		// }
	}
}


/*
 * calculate the signal power for Layers.LiHopfield
 */
void sig_power(double *res, const double *arr, int max_period, 
				int len, double th){
	double r;
	double sq_sum = 0;

	for (int j = 0; j < len; j++){
		res[j] = 0; // clear

		for (int i = 0; i < max_period; i++){
			r = arr[j * max_period + i];

			// count power
			if (r > th){
				res[j] += pow(r * 0.5, 2);
			}
		}

		// sum of squares
		sq_sum += pow(res[j], 2);
	}

	// if the power is significant
	if (sq_sum > 1e-15){ 
		// normalize
		for (int i = 0; i < len; i++){
			res[i] /= sqrt(sq_sum);

			RMNAN(&res[i]);
			// if (isnan(res[i])){
			// 	res[i] = 0;
			// }
		}
	}
}


/*
 * the depression function for Layers.BAM
 TODO: change the depression function
 */
void dep_f(double *w, const double *I, const int *shape, double phi){
	int size_in = shape[0];
	int size_out = shape[1];
	int k;
	double r, m;

	for (int i = 0; i < size_out; i++){
		for (int j = 0; j < size_in; j++){
			k = i + j * size_out;

			m = I[j] * pow(w[k], 2) * 50 + 1;

			if (m <= 0){
				continue;
			}

			r = 1 / m;

			if (r < 0){
				r = 0;
			}

			w[k] -= phi * r;

			RMNAN(&w[k]);
			// if (isnan(w[k])){
			// 	w[k] = 0;
			// }
		}
	}	
}


/*
 * the move function for Movement.RadMotor
 * Move to the next intended position
 */
void cmove(double *heading_rad, double *pos, const double preference,
			const double prev_preference, const double minStep, 
			const double lim, const double target_dir, const int sig_ign){
	double p = preference - prev_preference;

	// if no ignorinig signal
	if (!sig_ign){
		// make decision on where to go next
		// an alternative way (with more randomness) is disabled yet not discarded

		// srand((unsigned int) time(NULL));
		
		if (p > 1 && p / (abs(preference) + 1) > 0.05){
			// go forward
		
			*heading_rad = *heading_rad; 
			// *heading_rad = gaussian(*heading_rad, PI / (p * 200));
		
		} else if (p < -1 && -p / (abs(preference) + 1) > 0.05){
			// go backward
	    
	        *heading_rad = *heading_rad - PI; 
	        // *heading_rad = gaussian(*heading_rad - PI, PI / (-p * 200));
		
		} else{ // p in [-1, 1]
			// int rand_choice = rand() % 5;

			// if (rand_choice != 0 && preference != 0){ // random walk
	  		//	*heading_rad = rand() / (RAND_MAX + 1.0);
			// } else{ // go for the target
			*heading_rad = gaussian(target_dir, PI / 200);
			// }
		}

		// round
		if (*heading_rad > TWO_PI){
	    	*heading_rad = fmod(*heading_rad, TWO_PI);
		} else if (*heading_rad < 0){
			*heading_rad = fmod(*heading_rad, TWO_PI);
			*heading_rad += TWO_PI;
		}
	} 
	// else maintain the current heading rad

	// update increments
	double x = minStep * cos(*heading_rad);
	double y = minStep * sin(*heading_rad);

	// Actually move; not to exceed the boundary
	pos[0] = fmod((pos[0] + x), lim);
	pos[1] = fmod((pos[1] + y), lim);
}


/*
 * the judge function for Simulation.Simulation
 * Judge gustatory values and yield preference
 TODO: change/add judge functions
 */
void judge(double *p, double *to_judge, int *pfunc, int num_grn){
	const double sigma = 0.05;
	const double mu = 0.5;
	const double const_g = 10 / (sigma * sqrt(TWO_PI));

	double val;
	double res;

	for (int i = 0; i < num_grn; i++){
		val = to_judge[i];

		switch (pfunc[i]){
			case 0:
			// positive linear
				res = 100 * val;

				if (!isnan(res)){
					*p += res;
				}
				break;

			case 1:
			// negative linear
				res = -100 * val;
				
				if (!isnan(res)){
					*p += res;
				}
				break;

			case 2:
			// gaussian
				res = const_g * exp(- pow(((val - mu) / sigma), 2) / 2);
				
				if (!isnan(res)){
					*p += res;
				}
				break;
		}
	}

}