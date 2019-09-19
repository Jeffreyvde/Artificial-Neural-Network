using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public abstract class Activation
    {
        /// <summary>
        /// Calculate the activation function of the double
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public abstract double CalculateActivation(double value);

        /// <summary>
        /// Calculate the activation function for an entire vector
        /// </summary>
        /// <param name="value">Vector</param>
        /// <returns></returns>
        public abstract Vector<double> CalculateActivation(Vector<double> value);


        /// <summary>
        /// Calculate the derivative of the activation
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public abstract double CalculateDerivativeActivation(double value);

    }
}
