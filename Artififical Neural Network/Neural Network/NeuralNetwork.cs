using NeuralNetwork.Layers;
using System.Collections.Generic;
using NeuralNetwork.Backpropogation;
using System;

namespace NeuralNetwork.Basics
{
    [System.Serializable]
    public class NeuralNetwork
    {
        public InputLayer Input { get; private set; }
        public OutputLayer Output { get; private set; }

        private readonly Layer[] hiddenLayers;
        public IEnumerable<Layer> HiddenLayers => hiddenLayers;

        /// <summary>
        /// Constructor for the Neural Network
        /// </summary>
        public NeuralNetwork(InputLayer input, OutputLayer output, params Layer[] hidden)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            Output = output ?? throw new ArgumentNullException(nameof(output));
            hiddenLayers = hidden ?? throw new ArgumentNullException(nameof(hidden));

            List<Layer> layers = new List<Layer>()
            {
                input
            };
            layers.AddRange(hidden);
            layers.Add(output);
            for (int i = 0; i < layers.Count - 1; i++)
            {
                if (i == 0)
                    layers[i].ConnectNeurons(layers[i + 1].Neurons, null);
                else if (i == layers.Count - 1)
                    layers[i].ConnectNeurons(null, layers[i - 1].Neurons);
                else
                    layers[i].ConnectNeurons(layers[i - 1].Neurons, layers[i + 1].Neurons);
            }
        }

        /// <summary>
        /// Train the neural network
        /// </summary>
        public void FeedForward(double[] trainingData)
        {
            Input.SetInput(trainingData);
            for (int i = 1; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].FeedForward();
            }
            Output.FeedForward();
        }

        /// <summary>
        /// Back propagate neural network
        /// </summary>
        /// <param name="correctOutput">The expected output</param>
        public GradientDescent BackPropagate(double[] correctOutput)
        {
            GradientDescent gradient = new GradientDescent();
            Output.SetOutput(correctOutput);
            Output.BackPropagate(gradient);
            for (int i = hiddenLayers.Length; i > 0; i--)
            {
                hiddenLayers[i].BackPropagate(gradient);
            }
            Input.BackPropagate(gradient);
            return gradient;
        }
    }
}
