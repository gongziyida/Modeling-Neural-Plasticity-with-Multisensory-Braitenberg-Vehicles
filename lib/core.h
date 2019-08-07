#ifndef CORE_H
#define CORE_H

void sq_outer(double *res, const double *a1, const double *a2, 
				int len, char tri);

void I_minus_aL(double *L, double a, int size);

void rmnan(double *w, int *shape);

void set_stim_pos(int *pos, int num_stim, int max_pos, char method);

void build_space(double *space, const double *att, const int *pos, 
				int num_stim, int max_pos, int num_orn, 
				int pxl_dim, int gus_T);

void tanh_f(double *res, const double *arr, int len, int ax, double th);

void sig_power(double *res, const double *arr, int max_period, 
				int len, double th);

void dep_f(double *w, const double *I, const int *shape, double phi);

void cmove(double *heading_rad, double *pos, const double preference,
			const double prev_preference, const double minStep, 
			const double lim, const double target_dir);

void judge(double *preference, double *to_judge, int *pfunc, int num_grn);

#endif