namespace NeuralNetwork.Backpropogation
{
    public class GradientDecentValue
    {
        public double StepValue { get; set; }

        public IBackpropogatable ChangedObject { get; private set; }

        /// <summary>
        /// Create gradient descent value
        /// </summary>
        /// <param name="stepValue"></param>
        /// <param name="changedObject"></param>
        public GradientDecentValue(double stepValue, IBackpropogatable changedObject)
        {
            this.StepValue = stepValue;
            this.ChangedObject = changedObject;
        }

        /// <summary>
        /// Apply to the changed object
        /// </summary>
        /// <param name="learningRate"></param>
        public void Apply(double learningRate)
        {
            ChangedObject.ApplyGradientDecentStep(StepValue, learningRate);
        }
    }
}
