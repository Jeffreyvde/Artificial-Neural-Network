using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetworks
{
    public class Sigmoid : Activation
    {
        /// <summary>
        /// Calculate the sigmoid function of the double
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public override double CalculateActivation(double value)
        {
            double exponant = Math.Exp(value);
            return exponant / (1.0f + exponant);
        }

        /// <summary>
        /// Calculate the sigmoid function for an entire vector
        /// </summary>
        /// <param name="value">Vector</param>
        /// <returns></returns>
        public override Vector<double> CalculateActivation(Vector<double> value)
        {
            Vector<double> exponant = value.PointwiseExp();
            return exponant / (1.0f + exponant);
        }


        /// <summary>
        /// Calculate the derivative of sigmoid
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public override double CalculateDerivativeActivation(double value)
        {
            double derivativeExponant = 1 / (1 + Math.Exp(-value));
            return derivativeExponant * (1 - derivativeExponant);
        }
    }
}
