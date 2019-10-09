using Newtonsoft.Json;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class Layer
    {
        public readonly int index;

        public Weight[,] weights;
        private int weightRows, weightColumns;

        public readonly Neuron[] neurons;

        #region Initialization
        public Layer(int index, int sizeNeurons)
        {
            this.index = index;

            neurons = new Neuron[sizeNeurons];
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i] = new Neuron(index, i);
            }
        }

        [JsonConstructor()]
        public Layer(int index, Weight[,] weights, int weightRows, int weightColumns, Neuron[] neurons)
        {
            this.index = index;
            this.weights = weights;
            this.weightRows = weightRows;
            this.weightColumns = weightColumns;
            this.neurons = neurons;
        }

        /// <summary>
        /// Generate weights for this layer
        /// </summary>
        /// <param name="previousNeurons">previous neurons to create Connection</param>
        public void GenerateWeights(Neuron[] previousNeurons)
        {
            weightRows = neurons.Length;
            weightColumns = previousNeurons.Length;

            weights = new Weight[neurons.Length, previousNeurons.Length];
            for (int i = 0; i < neurons.Length; i++)
            {
                for (int j = 0; j < previousNeurons.Length; j++)
                {
                    weights[i, j] = new Weight(index, previousNeurons[j], neurons[i]);
                }
            }
        }
        #endregion
        #region Training

        /// <summary>
        /// Train this layer
        /// </summary>
        /// <param name="previousNeurons"></param>
        public void FeedForward(Neuron[] previousNeurons)
        {
            Matrix<double> weigthMatrix = Converter.ConvertToMatrix(weights, neurons.Length, previousNeurons.Length);
            Vector<double> activations = Converter.ConvertToVector(previousNeurons, true);
            Vector<double> biases = Converter.ConvertToVector(neurons, false);

            Vector<double> weightedSum = weigthMatrix * activations + biases;
            activations = NeuralNetwork.activation.CalculateActivation(weightedSum);
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].SetValues(weightedSum[i], activations[i]);
            }
        }

        #endregion
        #region Backpropogation

        /// <summary>
        /// Backpropogate this output layer
        /// </summary>
        /// <param name="correctOutputNeuron"></param>
        public void BackPropogate(int correctOutputNeuron, GradientDescent gradient)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].BackPropogate(correctOutputNeuron == i ? 1 : 0, gradient);
            }
            BackPropogateWeights(gradient);
        }

        /// <summary>
        /// Backpropogate this hidden layer
        /// </summary>
        /// <param name="nextLayer"></param>
        public void BackPropogate(Layer nextLayer, GradientDescent gradient)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].BackPropogate(nextLayer, gradient);
            }
            BackPropogateWeights(gradient);
        }

        /// <summary>
        /// Backpropogate all weights in this layer
        /// </summary>
        /// <param name="gradient"></param>
        private void BackPropogateWeights(GradientDescent gradient)
        {
            for (int x = 0; x < weightRows; x++)
            {
                for (int y = 0; y < weightColumns; y++)
                {
                    GetWeight(x, y).BackPropogate(gradient);
                }
            }
        }

        /// <summary>
        /// Get a weight
        /// </summary>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <returns></returns>
        public Weight GetWeight(int row, int column)
        {
            return weights[row, column];
        }

        #endregion


    }
}
