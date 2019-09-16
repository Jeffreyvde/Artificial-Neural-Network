using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetwork
{
    public static class Converter
    {

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
    }
}