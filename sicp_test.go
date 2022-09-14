package tycs_test

import (
	"fmt"
	"testing"
)

func sqrt_1_5(x float64) float64 {
	return sqrt_iter_1_5(1.0, x)
}

func sqrt_iter_1_5(guess, x float64) float64 {
	if guess_good_enough_1_5(guess, x) {
		return guess
	}

	return sqrt_iter_1_5(improve_guess(guess, x), x)
}

func guess_good_enough_1_5(guess, x float64) bool {
	sq := guess * guess
	abs := sq - x
	if abs < 0 {
		abs *= -1
	}
	return abs < 0.001
}

func improve_guess(guess, x float64) float64 {
	v := x / guess
	return (guess + v) / 2
}

func Test_sqrt_1_5(t *testing.T) {
	tt := []float64{2, 3}

	for _, tc := range tt {
		t.Run(fmt.Sprintf("%f", tc), func(t *testing.T) {
			fmt.Printf("%.3f\n", sqrt_1_5(tc))
		})
	}
}

// design a guess_good_enough function which tests the difference between current and previous guess.
func sqrt_1_7(x float64) float64 {
	return sqrt_iter_1_5(1.0, x)
}

func sqrt_iter_1_7(guess, prev_guess, x float64) float64 {
	if guess_good_enough_1_7(guess, prev_guess) {
		return guess
	}

	return sqrt_iter_1_7(improve_guess(guess, x), guess, x)
}

func guess_good_enough_1_7(guess, prev_guess float64) bool {
	v := guess - prev_guess
	if v < 0 {
		v *= -1
	}
	return v < 0.01
}

func Test_sqrt_1_7(t *testing.T) {
	tt := []float64{2, 3}

	for _, tc := range tt {
		t.Run(fmt.Sprintf("%f", tc), func(t *testing.T) {
			fmt.Printf("%.3f\n", sqrt_1_7(tc))
		})
	}
}
