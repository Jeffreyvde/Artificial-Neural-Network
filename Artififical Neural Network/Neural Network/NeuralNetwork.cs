using System;

namespace NeuralNetworks
{
    public class NeuralNetwork
    {
        public Layer[] layers;

        /// <summary>
        /// Create a neural network structure. The size include the input and output layers.
        /// </summary>
        /// <param name="layerSizes">sizes of each layer</param>
        public NeuralNetwork(params int[] layerSizes)
        {
            if (layerSizes.Length <= 2) throw new System.Exception("Neural network can not be smaller than 3 layers.");


            Random random = new Random();
            layers = new Layer[layerSizes.Length];

            CreateNewLayer(0, layerSizes[0], random, null);

            for (int i = 1; i < layerSizes.Length; i++)
            {
                CreateNewLayer(i, layerSizes[i], random, layers[i - 1].neurons);
            }

        }

        /// <summary>
        /// Train the neural network
        /// </summary>
        public void Train(TrainingData trainingData)
        {
            layers[0].AssingNeurons(trainingData.inputData);

            for (int i = 1; i < layers.Length; i++)
            {
                layers[1].Train(layers[i - 1].neurons);
            }
        }

        /// <summary>
        /// Create a new layer in the neural network
        /// </summary>
        /// <param name="index">What is the index of this new layer</param>
        /// <param name="size">What is the size of this new layer</param>
        /// <param name="random">Random as optimilization</param>
        /// <param name="previousNeurons">Optional previous neurons</param>
        private void CreateNewLayer(int index, int size, Random random, Neuron[] previousNeurons)
        {
            Layer layer = new Layer(index, size);
            layers[index] = layer;
            layer.GenerateNeurons(random);
            if (previousNeurons != null)
                layer.GenerateWeights(previousNeurons, random);
        }
    }
}
