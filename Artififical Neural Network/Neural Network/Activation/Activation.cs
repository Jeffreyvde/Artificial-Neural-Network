using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Activations
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
        /// Calculate the derivative of the activation
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        double CalculateDerivativeActivation(double value);

    }
}
