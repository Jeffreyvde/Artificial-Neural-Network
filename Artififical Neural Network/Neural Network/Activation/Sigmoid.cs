using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetwork
{
    public class Sigmoid : IActivation
    {
        /// <summary>
        /// Calculate the sigmoid function of the double
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double CalculateActivation(double value)
        {
            double exponant = Math.Exp(value);
            return exponant / (1.0f + exponant);
        }

        /// <summary>
        /// Calculate the sigmoid function for an entire vector
        /// </summary>
        /// <param name="value">Vector</param>
        /// <returns></returns>
        public Vector<double> CalculateActivation(Vector<double> value)
        {
            Vector<double> exponant = value.PointwiseExp();
            return exponant / (1.0f + exponant);
        }


        /// <summary>
        /// Calculate the derivative of sigmoid
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double CalculateDerivativeActivation(double value)
        {
            double derivativeExponant = 1 / (1 + Math.Exp(-value));
            return derivativeExponant * (1 - derivativeExponant);
        }
    }
}
