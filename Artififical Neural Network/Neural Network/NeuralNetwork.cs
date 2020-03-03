using NeuralNetwork.Layers;
using System.Collections.Generic;
using NeuralNetwork.Backpropogation;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace NeuralNetwork
{
    [System.Serializable]
    public class NeuralNetwork
    {
        public InputLayer Input { get; private set; }
        public OutputLayer Output { get; private set; }
        public Layer[] Hidden { get; private set; }

        /// <summary>
        /// Constructor for the Neural Network
        /// </summary>
        public NeuralNetwork(InputLayer input, OutputLayer output, params Layer[] hidden)
        {
            if (input == null || output == null) throw new System.ArgumentNullException("Input and output can not be null");

            this.Input = input;
            this.Output = output;
            this.Hidden = hidden;

            List<Layer> layers = new List<Layer>
            {
                input
            };
            for (int i = 0; i < hidden.Length; i++)
            {
                layers.Add(hidden[i]);
            }
            layers.Add(output);

            for (int i = 0; i < layers.Count; i++)
            {
                if (i == 0)
                    layers[i].ConnectNeurons(layers[i + 1].Neurons, null);
                else if (i == layers.Count - 1)
                    layers[i].ConnectNeurons(null, layers[i - 1].Neurons);
                else layers[i].ConnectNeurons(layers[i - 1].Neurons, layers[i + 1].Neurons);
            }
        }

        /// <summary>
        /// Train the neural network
        /// </summary>
        public void FeedForward(double[] trainingData)
        {
            Input.SetInput(trainingData);
            for (int i = 1; i < Hidden.Length; i++)
            {
                Hidden[i].FeedForward();
            }
            Output.FeedForward();
        }

        /// <summary>
        /// Backpropogate neural network
        /// </summary>
        /// <param name="trainingData"></param>
        public GradientDescent Backpropogate(double[] correctOuput)
        {
            GradientDescent gradient = new GradientDescent();
            Output.SetOutput(correctOuput);
            Output.BackPropogate(gradient);
            for (int i = Hidden.Length; i > 0; i--)
            {
                Hidden[i].BackPropogate(gradient);
            }
            Input.BackPropogate(gradient);
            return gradient;
        }

        /// <summary>
        /// Save neural network TODO Fix
        /// </summary>
        /// <param name="path"></param>
        public void Save(string path)
        {
            path = Combine(path);
            BinaryFormatter formatter = new BinaryFormatter();
            using (FileStream fileStream = File.Create(path))
            {
                formatter.Serialize(fileStream, this);
            }
        }

        /// <summary>
        /// Load Neural Network TODO Fix
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public static NeuralNetwork Load(string path)
        {
            path = Combine(path);
            if (!File.Exists(path))
                throw new System.ArgumentException("Path has no Neural Network");

            BinaryFormatter formatter = new BinaryFormatter();

            NeuralNetwork network;
            using (FileStream fileStream = File.Open(path, FileMode.Open))
            {
                network = (NeuralNetwork)formatter.Deserialize(fileStream);
            }
            return network;
        }

        private static string Combine(string path)
        {
            return path + "Neural.Network";
        }
    }
}
