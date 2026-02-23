/**
 * Computing Neuronal Synchrony.
 * Uses Kuramoto Order Parameter to compute syncrhonization.
 *
 * Compiled with: gcc -shared -o libkuramoto.so -fPIC kop.c -lm
 */

#include <complex.h> // complex, cexp, cabs, I
#include <math.h>    // M_PI, floor
#include <stdio.h>
#include <stdlib.h>

double *kuramoto_syn(double **sptime, double *t, double step_size,
                     double duration, int num_neurons,
                     int num_steps_simulation) {
    int num_steps = (int)(duration / step_size);

    // Preallocate phase array, fill with zeros.
    double **phi = (double **)malloc(num_steps_simulation * sizeof(double *));
    for (size_t i = 0; i < num_steps_simulation; i++) {
        phi[i] = (double *)calloc(num_neurons, sizeof(double));
    }

    // Preallocate array for resulting KOP values.
    double *result = (double *)malloc(num_steps_simulation * sizeof(double));

    // Calculate KOP values.
    for (size_t neuron = 0; neuron < num_neurons; neuron++) {
        int second_spike = 0;
        for (size_t i = 0; i < num_steps - 1; i++) {
            phi[i][neuron] = 2 * M_PI * (t[i] - sptime[i][neuron]);
            if (sptime[i + 1][neuron] != sptime[i][neuron]) {
                if (second_spike == 1) {
                    double delt = sptime[i + 1][neuron] - sptime[i][neuron];
                    int a = (int)floor(sptime[i][neuron] / step_size);
                    int b = (int)floor(sptime[i + 1][neuron] / step_size);
                    if (b >= num_steps_simulation) {
                        b = num_steps_simulation - 1;
                    }
                    if (delt != 0) {
                        for (size_t j = a; j <= b; j++) {
                            phi[j][neuron] /= delt;
                        }
                    }
                }
                second_spike = 1;
            }
        }
    }

    // Compute exponential sum.
    // R(t) * e^iψ(t) = 1/Ne * sum(e^iφk(t)), k from 1 to Ne.
    for (size_t i = 0; i < num_steps_simulation; i++) {
        double complex sum_exp = 0.0 + 0.0 * I;
        for (size_t neuron = 0; neuron < num_neurons; neuron++) {
            sum_exp += cexp(I * phi[i][neuron]);
        }
        result[i] = cabs(sum_exp / num_neurons);
    }

    for (size_t i = 0; i < num_steps_simulation; i++) {
        free(phi[i]);
    }
    free(phi);

    return result;
}