using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork.Neurons;

namespace NeuralNetwork.Utilities
{
    public static class Converter
    {
        /// <summary>
        /// Convert an array of weights to a Matrix
        /// </summary>
        /// <param name="neurons"></param>
        /// <returns></returns>
        public static Vector<double> GetWeights(Connection[] neurons)
        {
            if (neurons is null)
            {
                throw new System.ArgumentNullException(nameof(neurons));
            }

            Vector<double> values = Vector<double>.Build.Dense(neurons.Length);
            for (int i = 0; i < neurons.Length; i++)
            {
                values[i] = neurons[i].StartNeuron.Activation;
            }
            return values;
        }

        /// <summary>
        /// Convert an array of neurons to a Vector
        /// </summary>
        /// <param name="neurons"></param>
        /// <returns></returns>
        public static Vector<double> GetStartActivations(Connection[] neurons)
        {
            if (neurons is null)
            {
                throw new System.ArgumentNullException(nameof(neurons));
            }

            Vector<double> values = Vector<double>.Build.Dense(neurons.Length);
            for (int i = 0; i < neurons.Length; i++)
            {
                values[i] = neurons[i].StartNeuron.Activation;
            }
            return values;
        }
    }
}