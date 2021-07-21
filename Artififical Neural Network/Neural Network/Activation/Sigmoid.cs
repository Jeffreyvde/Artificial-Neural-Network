using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetwork.Activations
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
            double exponent = Math.Exp(value);
            return exponent / (1.0f + exponent);
        }

        /// <summary>
        /// Calculate the derivative of sigmoid
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double CalculateDerivativeActivation(double value)
        {
            double derivativeExponent = 1 / (1 + Math.Exp(-value));
            return derivativeExponent * (1 - derivativeExponent);
        }
    }
}
