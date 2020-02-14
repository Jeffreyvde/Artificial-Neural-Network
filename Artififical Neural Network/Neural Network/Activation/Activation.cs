using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    public interface IActivation
    {
        /// <summary>
        /// Calculate the activation function of the double
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        double CalculateActivation(double value);

        /// <summary>
        /// Calculate the activation function for an entire vector
        /// </summary>
        /// <param name="value">Vector</param>
        /// <returns></returns>
        Vector<double> CalculateActivation(Vector<double> value);


        /// <summary>
        /// Calculate the derivative of the activation
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        double CalculateDerivativeActivation(double value);

    }
}
