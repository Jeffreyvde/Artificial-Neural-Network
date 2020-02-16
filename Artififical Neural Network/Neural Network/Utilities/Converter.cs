using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork.Neurons;

namespace NeuralNetwork
{
    public static class Converter
    {
        /// <summary>
        /// Convert an array of weights to a Matrix<double>
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="rows"></param>
        /// <param name="layers"></param>
        /// <returns></returns>
        public static Vector<double> GetWeights(Connection[] neurons)
        {
            Vector<double> values = Vector<double>.Build.Dense(neurons.Length);
            for (int i = 0; i < neurons.Length; i++)
            {
                values[i] = neurons[i].StartNeuron.Activation;
            }
            return values;
        }

        /// <summary>
        /// Convert an array of neurons to a Vector<double>
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="rows"></param>
        /// <param name="layers"></param>
        /// <returns></returns>
        public static Vector<double> GetStartActivations(Connection[] neurons)
        {
            Vector<double> values = Vector<double>.Build.Dense(neurons.Length);
            for (int i = 0; i < neurons.Length; i++)
            {
                values[i] = neurons[i].StartNeuron.Activation;
            }
            return values;
        }
    }
}