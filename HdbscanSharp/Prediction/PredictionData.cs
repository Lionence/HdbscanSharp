using System.Collections.Generic;

namespace HdbscanSharp.Prediction
{
    public class PredictionData
    {
        public double[][] RawData { get; set; }
        public double[] CoreDistances { get; set; }
        public int MinSamples { get; set; }

        public int[] Labels { get; set; }

        public int[] TreeParent { get; set; }
        public int[] TreeChild { get; set; }
        public double[] TreeLambdaVal { get; set; }
        public int[] TreeChildSize { get; set; }

        public int[] ClusterTreeParent { get; set; }
        public int[] ClusterTreeChild { get; set; }
        public double[] ClusterTreeLambdaVal { get; set; }
        public int[] ClusterTreeChildSize { get; set; }

        public Dictionary<int, int> ClusterMap { get; set; }
        public Dictionary<int, double> MaxLambdas { get; set; }
    }
}