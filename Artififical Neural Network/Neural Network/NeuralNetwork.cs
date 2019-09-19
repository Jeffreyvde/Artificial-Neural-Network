using System;
using System.Linq;

namespace NeuralNetworks
{
    public class NeuralNetwork
    {
        public Layer[] layers;
        private readonly IActivation activation;

        /// <summary>
        /// Create a neural network structure. The size include the input and output layers.
        /// </summary>
        /// <param name="layerSizes">sizes of each layer</param>
        public NeuralNetwork(IActivation activation, params int[] layerSizes)
        {
            if (layerSizes.Length <= 2) throw new System.Exception("Neural network can not be smaller than 3 layers.");

            this.activation = activation;

            layers = new Layer[layerSizes.Length];

            CreateNewLayer(0, layerSizes[0], null);

            for (int i = 1; i < layerSizes.Length; i++)
            {
                CreateNewLayer(i, layerSizes[i], layers[i - 1].neurons);
            }
        }

        /// <summary>
        /// Train the neural network
        /// </summary>
        public void Train(TrainingData trainingData)
        {
            SetInputLayer(trainingData.inputData);

            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].Train(layers[i - 1].neurons);
            }
        }

        /// <summary>
        /// Backpropogate neural network
        /// </summary>
        /// <param name="trainingData"></param>
        public GradientDescent Backpropogate(TrainingData trainingData)
        {
            GradientDescent gradient = new GradientDescent();
            layers[layers.Length - 1].BackPropogate(trainingData.correctOutputNeuron, gradient);

            for (int i = layers.Length - 2; i > 0; i--)
            {
                layers[i].BackPropogate(layers[i + 1], gradient);
            }
            return gradient;
        }

        /// <summary>
        /// Create a new layer in the neural network
        /// </summary>
        /// <param name="index">What is the index of this new layer</param>
        /// <param name="size">What is the size of this new layer</param>
        /// <param name="random">Random as optimilization</param>
        /// <param name="previousNeurons">Optional previous neurons</param>
        private void CreateNewLayer(int index, int size, Neuron[] previousNeurons)
        {
            Layer layer = new Layer(index, size, activation);
            layers[index] = layer;
            if (previousNeurons != null)
                layer.GenerateWeights(previousNeurons);
        }

        /// <summary>
        /// Initialize the input layer
        /// </summary>
        /// <param name="input"></param>
        private void SetInputLayer(double[] input)
        {
            Layer layer = layers[0];

            if (input.Length != layer.neurons.Length) throw new Exception("Input not equal to neurons lenght");

            for (int i = 0; i < layer.neurons.Length; i++)
            {
                layer.neurons[i] = new Neuron(0, i, input[i]);
            }
        }
    }
}
