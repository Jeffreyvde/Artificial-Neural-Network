using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetworks
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
        public static Matrix<double> ConvertToMatrix(Weight[,] weights, int rows, int layers)
        {
            double[,] weightValues = new double[rows, layers];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < layers; j++)
                {
                    weightValues[i, j] = weights[i, j].weight;
                }
            }
            return DenseMatrix.OfArray(weightValues);
        }

        /// <summary>
        /// Convert an array of neurons to a Vector<double>
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="rows"></param>
        /// <param name="layers"></param>
        /// <returns></returns>
        public static Vector<double> ConvertToVector(Neuron[] neurons, bool activationVector)
        {
            Vector<double> values = Vector<double>.Build.Dense(neurons.Length);
            for (int i = 0; i < neurons.Length; i++)
            {
                values.Add(activationVector ? neurons[i].activation : neurons[i].bias);
            }
            return values;
        }
    }
}