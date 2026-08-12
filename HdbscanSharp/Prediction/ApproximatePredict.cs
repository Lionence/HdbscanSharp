#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using HdbscanSharp.Hdbscanstar;
using HdbscanSharp.Runner;

namespace HdbscanSharp.Prediction
{
    public static class ApproximatePredict
    {
        public static PredictionData BuildPredictionData(
            double[][] trainingData,
            int minPoints,
            int minClusterSize,
            Func<int, int, double> distanceFunc,
            List<HdbscanConstraint> constraints = null)
        {
            var numPoints = trainingData.Length;

            var coreDistances = HdbscanAlgorithm.CalculateCoreDistances(
                distanceFunc, numPoints, minPoints);

            var mst = HdbscanAlgorithm.ConstructMst(
                distanceFunc, numPoints, coreDistances, true);
            mst.QuicksortByEdgeWeight();

            var pointNoiseLevels = new double[numPoints];
            var pointLastClusters = new int[numPoints];
            var hierarchy = new List<int[]>();

            var clusters = HdbscanAlgorithm.ComputeHierarchyAndClusterTree(
                mst, minClusterSize, constraints, hierarchy,
                pointNoiseLevels, pointLastClusters);

            HdbscanAlgorithm.PropagateTree(clusters);

            var labels = HdbscanAlgorithm.FindProminentClusters(
                clusters, hierarchy, numPoints);

            // Build condensed tree
            var treeParent = new List<int>();
            var treeChild = new List<int>();
            var treeLambdaVal = new List<double>();
            var treeChildSize = new List<int>();

            foreach (var cluster in clusters)
            {
                if (cluster == null || cluster.Label < 2)
                    continue;
                treeParent.Add(cluster.Parent?.Label ?? 1);
                treeChild.Add(cluster.Label);
                treeLambdaVal.Add(cluster.BirthLevel > 0
                    ? 1.0 / cluster.BirthLevel
                    : double.MaxValue);
                treeChildSize.Add(cluster.InitialNumPoints);
            }

            for (int p = 0; p < numPoints; p++)
            {
                if (pointNoiseLevels[p] > 0)
                {
                    treeParent.Add(pointLastClusters[p]);
                    treeChild.Add(p);
                    treeLambdaVal.Add(1.0 / pointNoiseLevels[p]);
                    treeChildSize.Add(1);
                }
            }

            // Build cluster tree (subset where child_size > 1)
            var clusterTreeParent = new List<int>();
            var clusterTreeChild = new List<int>();
            var clusterTreeLambdaVal = new List<double>();
            var clusterTreeChildSize = new List<int>();

            for (int i = 0; i < treeParent.Count; i++)
            {
                if (treeChildSize[i] > 1)
                {
                    clusterTreeParent.Add(treeParent[i]);
                    clusterTreeChild.Add(treeChild[i]);
                    clusterTreeLambdaVal.Add(treeLambdaVal[i]);
                    clusterTreeChildSize.Add(treeChildSize[i]);
                }
            }

            // Build cluster map
            var selectedClusters = clusters[1].PropagatedDescendants;
            var clusterMap = new Dictionary<int, int>();
            var nextLabel = 0;

            foreach (var selected in selectedClusters)
            {
                clusterMap[selected.Label] = nextLabel;

                var toProcess = new Queue<int>();
                toProcess.Enqueue(selected.Label);

                while (toProcess.Count > 0)
                {
                    var current = toProcess.Dequeue();
                    for (int i = 0; i < clusterTreeParent.Count; i++)
                    {
                        if (clusterTreeParent[i] == current)
                        {
                            var child = clusterTreeChild[i];
                            if (!clusterMap.ContainsKey(child))
                            {
                                clusterMap[child] = nextLabel;
                                toProcess.Enqueue(child);
                            }
                        }
                    }
                }

                nextLabel++;
            }

            clusterMap[0] = -1;

            // Build max lambdas
            var maxLambdas = new Dictionary<int, double>();
            foreach (var cluster in clusters)
            {
                if (cluster == null)
                    continue;
                var deathLevel = cluster.PropagatedLowestChildDeathLevel;
                maxLambdas[cluster.Label] = deathLevel > 0 && deathLevel < double.MaxValue
                    ? 1.0 / deathLevel
                    : double.MaxValue;
            }

            return new PredictionData
            {
                RawData = trainingData,
                CoreDistances = coreDistances,
                MinSamples = minPoints,
                Labels = labels,
                TreeParent = treeParent.ToArray(),
                TreeChild = treeChild.ToArray(),
                TreeLambdaVal = treeLambdaVal.ToArray(),
                TreeChildSize = treeChildSize.ToArray(),
                ClusterTreeParent = clusterTreeParent.ToArray(),
                ClusterTreeChild = clusterTreeChild.ToArray(),
                ClusterTreeLambdaVal = clusterTreeLambdaVal.ToArray(),
                ClusterTreeChildSize = clusterTreeChildSize.ToArray(),
                ClusterMap = clusterMap,
                MaxLambdas = maxLambdas
            };
        }

        public static (int[] labels, double[] probabilities) Predict(
            PredictionData data,
            double[][] pointsToPredict,
            Action<string>? trace = null)
        {
            var numPoints = pointsToPredict.Length;
            var labels = new int[numPoints];
            var probabilities = new double[numPoints];
            var minSamples = data.MinSamples;
            var k = 15;

            var clusterTreeLookup = new Dictionary<int, (int parent, double lambdaVal)>();
            for (int i = 0; i < data.ClusterTreeChild.Length; i++)
            {
                clusterTreeLookup[data.ClusterTreeChild[i]] = (
                    data.ClusterTreeParent[i],
                    data.ClusterTreeLambdaVal[i]);
            }
            var treeRoot = data.ClusterTreeParent.Length > 0
                ? data.ClusterTreeParent.Min()
                : 0;

            for (int p = 0; p < numPoints; p++)
            {
                var point = pointsToPredict[p];

                var neighborDistances = new double[data.RawData.Length];
                for (int i = 0; i < data.RawData.Length; i++)
                {
                    neighborDistances[i] = CosineDistance(point, data.RawData[i]);
                }

                var neighborIndices = Enumerable.Range(0, data.RawData.Length)
                    .OrderBy(i => neighborDistances[i])
                    .Take(k)
                    .ToArray();

                var nearestDistances = neighborIndices.Select(i => neighborDistances[i]).ToArray();

                var pointCoreDistance = nearestDistances.Length > 0
                    ? nearestDistances[Math.Min(nearestDistances.Length - 1, minSamples > 0 ? minSamples - 1 : 0)]
                    : double.MaxValue;

                var treeLookup = new Dictionary<int, (int parent, double lambdaVal)>();
                for (int i = 0; i < data.TreeChild.Length; i++)
                {
                    treeLookup[data.TreeChild[i]] = (data.TreeParent[i], data.TreeLambdaVal[i]);
                }

                var voteCounts = new Dictionary<int, int>();
                var totalVotes = 0;

                for (int n = 0; n < neighborIndices.Length; n++)
                {
                    var neighborIdx = neighborIndices[n];
                    var mrDist = Math.Max(
                        data.CoreDistances[neighborIdx],
                        Math.Max(pointCoreDistance, nearestDistances[n]));

                    var lambda = mrDist > 0 ? 1.0 / mrDist : double.MaxValue;

                    var potentialCluster = -1;
                    if (treeLookup.TryGetValue(neighborIdx, out var neighborRow))
                    {
                        potentialCluster = neighborRow.parent;
                        var neighborLambda = neighborRow.lambdaVal;

                        if (neighborLambda > lambda)
                        {
                            while (potentialCluster > treeRoot)
                            {
                                if (clusterTreeLookup.TryGetValue(potentialCluster, out var entry))
                                {
                                    if (entry.lambdaVal >= lambda)
                                        potentialCluster = entry.parent;
                                    else
                                        break;
                                }
                                else
                                {
                                    break;
                                }
                            }
                        }
                    }

                    if (potentialCluster < 0)
                    {
                        potentialCluster = 1;
                    }

                    if (data.ClusterMap.TryGetValue(potentialCluster, out var mappedLabel) && mappedLabel >= 0)
                    {
                        voteCounts.TryGetValue(mappedLabel, out var current);
                        voteCounts[mappedLabel] = current + 1;
                        totalVotes++;
                    }
                }

                if (totalVotes > 0)
                {
                    var winner = voteCounts.OrderByDescending(kv => kv.Value).First();
                    labels[p] = winner.Key;
                    probabilities[p] = (double)winner.Value / totalVotes;
                }
                else
                {
                    labels[p] = -1;
                    probabilities[p] = 0.0;
                }

                var voteDetails = string.Join(", ", voteCounts.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}:{kv.Value}"));
                trace?.Invoke($"  chunk[{p}]: kNN votes=[{voteDetails}] winner={labels[p]} prob={probabilities[p]:G4}");
            }

            return (labels, probabilities);
        }

        public static (string label, double score, string decision) PredictDocument(
            PredictionData data,
            double[][] chunkEmbeddings,
            double autoAcceptThreshold = 0.95,
            double underReviewThreshold = 0.70,
            Action<string>? trace = null)
        {
            trace?.Invoke($"PredictDocument: {chunkEmbeddings.Length} chunks, autoAcceptThreshold={autoAcceptThreshold}, underReviewThreshold={underReviewThreshold}");
            var (labels, probs) = Predict(data, chunkEmbeddings, trace);

            var evidence = new Dictionary<int, double>();
            for (int i = 0; i < labels.Length; i++)
            {
                if (labels[i] >= 0)
                {
                    if (!evidence.TryGetValue(labels[i], out var existing))
                        existing = 0;
                    evidence[labels[i]] = existing + probs[i];
                }
            }

            if (evidence.Count == 0)
            {
                trace?.Invoke("PredictDocument: evidence is empty (all chunks classified as noise) -> returning (-1, 1.0, under_review)");
                return ("-1", 1.0, "under_review");
            }

            var winner = evidence.OrderByDescending(kv => kv.Value).First();
            var score = winner.Value / chunkEmbeddings.Length;

            trace?.Invoke($"PredictDocument: evidence=[{string.Join(", ", evidence.Select(kv => $"{kv.Key}:{kv.Value:G4}"))}]");
            trace?.Invoke($"PredictDocument: winner=clusterLabel {winner.Key}, evidenceSum={winner.Value:G4}, score={score:G4}");

            string decision;
            string label;
            if (score < underReviewThreshold)
            {
                label = "-1";
                decision = "under_review";
                trace?.Invoke($"PredictDocument: score < {underReviewThreshold} -> label=-1, under_review");
            }
            else if (score >= autoAcceptThreshold)
            {
                label = winner.Key.ToString();
                decision = "auto_classified";
                trace?.Invoke($"PredictDocument: score >= {autoAcceptThreshold} -> label={label}, auto_classified");
            }
            else
            {
                label = winner.Key.ToString();
                decision = "under_review";
                trace?.Invoke($"PredictDocument: score in [{underReviewThreshold}, {autoAcceptThreshold}) -> label={label}, under_review");
            }

            return (label, score, decision);
        }

        private static double CosineDistance(double[] a, double[] b)
        {
            double dot = 0, normA = 0, normB = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            normA = Math.Sqrt(normA);
            normB = Math.Sqrt(normB);
            if (normA == 0 || normB == 0)
                return 1.0;
            return 1.0 - (dot / (normA * normB));
        }
    }
}