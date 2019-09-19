namespace NeuralNetworks
{
    public class NeuralNetwork
    {
        public Layer[] layers;
        private IActivation activation;


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
            layers[0].AssingNeurons(trainingData.inputData);

            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].Train(layers[i - 1].neurons);
            }
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
            layer.GenerateNeurons();
            if (previousNeurons != null)
                layer.GenerateWeights(previousNeurons);
        }
    }
}
