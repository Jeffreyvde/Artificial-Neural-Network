namespace NeuralNetworks
{
    public class TrainingData
    {
        public byte[] inputData;
        public byte correctOutputNeuron;

        public TrainingData(byte[] input, byte correctOuput)
        {
            inputData = input;
            correctOutputNeuron = correctOuput;
        }
    }
}
