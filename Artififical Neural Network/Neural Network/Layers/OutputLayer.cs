using System;
using NeuralNetwork.Neurons;
using NeuralNetwork.Activations;

namespace NeuralNetwork.Layers
{
    [Serializable]
    public class OutputLayer : Layer
    {
        /// <summary>
        /// Initialise the Layer with Hidden neurons
        /// </summary>
        /// <param name="size">The size of this layer</param>
        /// <param name="activationFunction">The function to be used for the activation calculations</param>
        public OutputLayer(int size, IActivation activationFunction)
        {
            if(activationFunction == null)
                throw new  ArgumentNullException(nameof(activationFunction));
            Neurons = new BaseNeuron[size];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new OutPutNeuron(activationFunction);
            }
        }

        /// <summary>
        /// Set the input data for the input neurons
        /// </summary>
        /// <param name="correctOutput">The correct output for this Neuron</param>
        public void SetOutput(double[] correctOutput)
        {
            if (correctOutput == null) 
                throw new ArgumentNullException(nameof(correctOutput));
            if (correctOutput.Length != Neurons.Length) 
                throw new ArgumentException($"{nameof(correctOutput)} can not be a different length from the neurons of this layer: Please check that output data is {Neurons.Length} long.");

            for (int i = 0; i < Neurons.Length; i++)
            {
                ((OutPutNeuron)Neurons[i]).SetOutput(correctOutput[i]);
            }
        }
    }
}
