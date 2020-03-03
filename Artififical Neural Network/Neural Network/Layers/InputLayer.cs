using NeuralNetwork.Neurons;

namespace NeuralNetwork.Layers
{
    public class InputLayer : Layer
    {
        /// <summary>
        /// Initialize from size
        /// </summary>
        /// <param name="size"></param>
        /// <param name="activationFunction"></param>
        public InputLayer(int size)
        {
            Neurons = new InputNeuron[size];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new InputNeuron();
            }
        }

        /// <summary>
        /// Initialize from training data
        /// </summary>
        /// <param name="startData"></param>
        /// <param name="activationFunction"></param>
        public InputLayer(double[] startData)
        {
            if (startData == null) throw new System.ArgumentNullException("Startdata can not be null: please check Input layer constructor");

            Neurons = new InputNeuron[startData.Length];
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
            if (startData == null) throw new System.ArgumentException("Startdata can not be null: please check Input layer's SetInput function");
            else if (startData.Length != Neurons.Length) throw new System.ArgumentException("Startdata can not be a different length from the Neurons of this layer: Please check input data or layer initialization.");
 
            for (int i = 0; i < Neurons.Length; i++)
            {
                ((InputNeuron)Neurons[i]).SetActivation(startData[i]);
            }
        }
    }
}
