using System;
using System.IO;
using System.Text;
using Newtonsoft.Json;

namespace NeuralNetworks
{
    public class NeuralNetwork
    {
        public Layer[] layers;
        public static IActivation activation;

        /// <summary>
        /// Create a neural network structure. The size include the input and output layers.
        /// </summary>
        /// <param name="layerSizes">sizes of each layer</param>
        public NeuralNetwork(IActivation activation, params int[] layerSizes)
        {
            if (layerSizes.Length <= 2) throw new System.Exception("Neural network can not be smaller than 3 layers.");

            NeuralNetwork.activation = activation;

            layers = new Layer[layerSizes.Length];

            CreateNewLayer(0, layerSizes[0], null);

            for (int i = 1; i < layerSizes.Length; i++)
            {
                CreateNewLayer(i, layerSizes[i], layers[i - 1].neurons);
            }
        }

        [JsonConstructor()]
        public NeuralNetwork(Layer[] layers)
        {
            this.layers = layers;
        }

        /// <summary>
        /// Train the neural network
        /// </summary>
        public void FeedForward(TrainingData trainingData)
        {
            SetInputLayer(trainingData.inputData);

            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].FeedForward(layers[i - 1].neurons);
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
            Layer layer = new Layer(index, size);
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
                layer.neurons[i].activation = input[i];
            }
        }

        public double CalculateCost(int correctOuput)
        {
            Layer lastLayer = layers[layers.Length - 1];

            double cost = 0;
            for (int i = 0; i < lastLayer.neurons.Length; i++)
            {
                cost += lastLayer.neurons[i].CalculateCost(correctOuput);
            }
            return cost / lastLayer.neurons.Length;
        }

        public bool IsNeuralNetworkCorrect(int correctOutput)
        {
            Layer lastLayer = layers[layers.Length - 1];

            int guess = 0;
            double highestActivation = 0;
            for (int i = 0; i < lastLayer.neurons.Length; i++)
            {
                double activation = lastLayer.neurons[i].activation;
                if (highestActivation < activation)
                {
                    guess = i;
                    highestActivation = activation;
                }
            }
            return guess == correctOutput;
        }


        public void Save(string path)
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
            using (FileStream fs = File.Create(path))
            {
                Byte[] info = new UTF8Encoding(true).GetBytes(JsonConvert.SerializeObject(this, new JsonSerializerSettings { PreserveReferencesHandling = PreserveReferencesHandling.Objects }));
                // Add some information to the file.
                fs.Write(info, 0, info.Length);
            }
        }

        public static NeuralNetwork Load(string path, IActivation activation)
        {
            using (StreamReader r = new StreamReader(path))
            {
                string json = r.ReadToEnd();
                NeuralNetwork value = JsonConvert.DeserializeObject<NeuralNetwork>(json);
                NeuralNetwork.activation = activation;
                return value;
            }
        }
    }
}
