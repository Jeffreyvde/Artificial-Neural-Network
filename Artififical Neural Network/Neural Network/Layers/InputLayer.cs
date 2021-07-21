using NeuralNetwork.Neurons;
using System;

namespace NeuralNetwork.Layers
{
    [System.Serializable]
    public class InputLayer : Layer
    {
        /// <summary>
        /// Initialise the Layer with Hidden neurons
        /// </summary>
        /// <param name="size">The size of this layer</param>
        public InputLayer(uint size)
        {
            Neurons = new BaseNeuron[size];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new InputNeuron();
            }
        }

        /// <summary>
        /// Initialize from training data
        /// </summary>
        /// <param name="startData">The starting data you want for this Neuron</param>
        public InputLayer(double[] startData)
        {
            if (startData == null) 
                throw new ArgumentNullException(nameof(startData));

            Neurons = new BaseNeuron[startData.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new InputNeuron(startData[i]);
            }
        }

        /// <summary>
        /// Set the input data for the input neurons
        /// </summary>
        /// <param name="startData"></param>
        public void SetInput(double[] startData)
        {
            if (startData == null) throw new ArgumentNullException(nameof(startData));
            if (startData.Length != Neurons.Length) throw new ArgumentException($"{nameof(startData)} can not be a different length from the Neurons of this layer: Please check input data or layer initialization."); 

            for (int i = 0; i < Neurons.Length; i++)
            {
                ((InputNeuron)Neurons[i]).SetActivation(startData[i]);
            }
        }
    }
}
