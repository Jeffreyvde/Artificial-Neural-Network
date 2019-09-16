using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetworks
{
    public static class Sigmoid
    {
        /// <summary>
        /// Calculate the sigmoid function of the double
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double CalculateSigmoid(double value)
        {
            double exponant = Math.Exp(value);
            return exponant / (1.0f + exponant);
        }

        /// <summary>
        /// Calculate the sigmoid function for an entire vector
        /// </summary>
        /// <param name="value">Vector</param>
        /// <returns></returns>
        public static Vector<double> CalculateSigmoid(Vector<double> value)
        {
            Vector<double> exponant = value.PointwiseExp();
            return exponant / (1.0f + exponant);
        }


        /// <summary>
        /// Calculate the derivative of sigmoid
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double CaclulateDerivativeSigmoid(double value)
        {
            double derivativeExponant = 1 / (1 + Math.Exp(-value));
            return derivativeExponant * (1 - derivativeExponant);
        }

    }
}
