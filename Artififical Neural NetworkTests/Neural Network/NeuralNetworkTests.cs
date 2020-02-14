using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void LayerWeightsInitialized()
        {
            NeuralNetwork neuralNetwork = new NeuralNetwork(new Sigmoid(), 3, 3, 3);

            for (int i = 1; i < neuralNetwork.layers.Length; i++)
            {

                Assert.IsNotNull(neuralNetwork.layers[i].weights, "Weights of network layers not initialized");

            }
        }

        [TestMethod()]
        public void LayerNeuronsInitialized()
        {
            NeuralNetwork neuralNetwork = new NeuralNetwork(new Sigmoid(), 3, 3, 3);

            for (int i = 0; i < neuralNetwork.layers.Length; i++)
            {
                Assert.IsNotNull(neuralNetwork.layers[i].neurons, "Neurons of network layes not initialized");
            }
        }
    }
}