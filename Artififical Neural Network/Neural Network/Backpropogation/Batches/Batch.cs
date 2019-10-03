namespace NeuralNetworks
{
    public class Batch
    {
        public TrainingData[] data;
        public GradientDescent descent;

        public Batch(TrainingData[] allData, int size)
        {

            data = new TrainingData[size];
            for(int i = 0; i < size; i++)
            {
                data[i] = allData[Randomizer.Range(0, allData.Length)];
            }

        }
    }
}
