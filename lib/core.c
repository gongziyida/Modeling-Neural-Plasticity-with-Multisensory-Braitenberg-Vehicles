#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#define PI 3.141592654
#define TWO_PI 3.141592654 * 2


/*************************************************************
 * Gaussian random value generator
 *************************************************************/
double gaussian(double m, double std){
	static double U, V;
	static int phase = 0;
	double z;

	if (phase == 0){
		U = rand() / (RAND_MAX + 1.0);
		V = rand() / (RAND_MAX + 1.0);
		z = sqrt(-2.0 * log(U))* sin(2.0 * PI * V);
	} else{
		z = sqrt(-2.0 * log(U)) * cos(2.0 * PI * V);
	}

	phase = 1 - phase;
	z = z * std + m;
	return z;
}


/*************************************************************
 * calculate the outer product, or the triangular part of 
 * the outer product
 *************************************************************/
void sq_outer(double *res, const double *a1, const double *a2, 
				int len, char tri){
	if (tri == 'u' || tri == 'n'){
		for (int j = 0; j < len; j++){
			for (int i = j; i < len; i++){
				res[i * len + j] = a1[j] * a2[i];
			}
		}
	} 

	if (tri == 'l' || tri == 'n'){
		for (int i = 0; i < len; i++){
			for (int j = i; j < len; j++){
				res[i * len + j] = a1[j] * a2[i];
			}
		}
	}
}


/*************************************************************
 * calculate I - a * L, where I is an identity matrix and a is a scalar
 *************************************************************/
void I_minus_aL(double *L, double a, int size){
	int k;
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			k = i + j * size;
			if (j == i){
				L[k] = 1 - a * L[k];
			} else{
				L[k] = - a * L[k];
			}
		}
	}
}


// Following are the functions specified for certain classes

/*************************************************************
 * Select positions of the stimulus sources for class Space.Space 
 *************************************************************/
void set_stim_pos(int *pos, int num_stim, int max_pos, char method){
	if (method == 'r'){
		srand((unsigned int) time(NULL));
	} 

	int r = (int) sqrt(num_stim);
	int c = num_stim / r;
	int rmd = num_stim - r * c;

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
			
			if (method == 'r'){
				pos[i] += rand() % xitvl;
				pos[i+1] += rand() % yitvl;
			}
    	}
   	}
}


/*************************************************************
 * Build the space for class Space.Space 
 *************************************************************/
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
				xdiff = diff(pos[k * 2] - i, max_pos_d);
				ydiff = diff(pos[k * 2 + 1] - j, max_pos_d);
				dist = pow(xdiff, 2);
				dist += pow(ydiff, 2);
				dist = sqrt(dist);

				factor = exp(-dist / max_pos_d * 10);

				for (int a = 0; a < num_orn; a++){
					si = (i * max_pos + j) * pxl_dim + a;
					ai = k * pxl_dim + a;
					space[si] += factor * att[ai];

					if (k == num_stim - 1 && space[si] < 0.0001){
						space[si] = 0;
					}
				}

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


/*************************************************************
 * the tanh function for Layers.LiHopfield
 *************************************************************/
void tanh_f(double *res, const double *arr, int len, int ax, double th){
	double s, s_, r, factor;
	if (ax == 0){
		s_ = 0.14;
	} else if (ax == 1){
		s_ = 0.29;
	}

	for (int i = 0; i < len; i++){
		r = arr[i] - th;

		if (r < 0){
			s = s_;
		} else{
			s = s_ * 10;
		}

		res[i] = s_ + s * tanh(r / s);
	}
}


/*************************************************************
 * calculate the signal power for Layers.LiHopfield
 *************************************************************/
void sig_power(double *res, const double *arr, int len, double th){
	double r;
	for (int i = 0; i < len; i++){
		r = arr[i] - th;

		if (r > 0){
			res[i] += pow(arr[i] * 0.5, 2);
		}
	}
}


/*************************************************************
 * the depression function for Layers.BAM
 *************************************************************/
void dep_f(double *w, const double *I, const int *shape, double phi){
	int size_in = shape[0];
	int size_out = shape[1];
	int k;

	for (int i = 0; i < size_out; i++){
		for (int j = 0; j < size_in; j++){
			k = i + j * size_out;
			w[k] -= phi / (I[j] * pow(w[k], 2) + phi);
		}
	}	
}


/*************************************************************
 * the move function for Movement.RadMotor
 *************************************************************/
void cmove(double *heading_rad, double *pos, const double p, 
			const double minStep, const double lim, 
			const double target_dir){
	
	// make decision on where to go next
	srand((unsigned int) time(NULL));
	if (p > 1){
		// go forward
		*heading_rad = gaussian(*heading_rad, PI / (p * 2));
	} else if (p < -1){
		// go backward
        *heading_rad = gaussian(*heading_rad - PI, PI / (-p * 2));
	} else{ // p in [-1, 1]
		int rand_choice = rand() % 2;

		if (rand_choice){ // random walk
        	*heading_rad = rand() / (RAND_MAX + 1.0);
		} else{ // go for the target
			*heading_rad = target_dir;
		}
	}

	// round
	if (*heading_rad > TWO_PI){
    	*heading_rad = fmod(*heading_rad, TWO_PI);
	} else if (*heading_rad < 0){
		*heading_rad = fmod(*heading_rad, TWO_PI);
		*heading_rad += TWO_PI;
	}

	// update increments
	double x = minStep * cos(*heading_rad);
	double y = minStep * sin(*heading_rad);

	// Actually move; not to exceed the boundary
	pos[0] = fmod((pos[0] + x), lim);
	pos[1] = fmod((pos[1] + y), lim);
}


