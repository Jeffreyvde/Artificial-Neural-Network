namespace NeuralNetwork
{
    public class GradientDecentValue
    {
        private double stepValue;

        private IBackpropogatable changedObject;

        /// <summary>
        /// Create gradient descent value
        /// </summary>
        /// <param name="stepValue"></param>
        /// <param name="changedObject"></param>
        public GradientDecentValue(double stepValue, IBackpropogatable changedObject)
        {
            this.stepValue = stepValue;
            this.changedObject = changedObject;
        }

        /// <summary>
        /// Apply to the changed object
        /// </summary>
        /// <param name="learningRate"></param>
        public void Apply(double learningRate)
        {
            changedObject.ApplyGradientDecentStep(stepValue, learningRate);
        }
    }
}
