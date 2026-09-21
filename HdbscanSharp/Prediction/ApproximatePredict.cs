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
            Action<string>? trace = null,
            int neighborCount = 15)
        {
            var numPoints = pointsToPredict.Length;
            var labels = new int[numPoints];
            var probabilities = new double[numPoints];
            var minSamples = data.MinSamples;
            var k = neighborCount;

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
            IReadOnlyDictionary<string, string>? categoryMap = null,
            double autoAcceptThreshold = 0.95,
            double underReviewThreshold = 0.70,
            Action<string>? trace = null,
            int neighborCount = 15)
        {
            trace?.Invoke($"PredictDocument: {chunkEmbeddings.Length} chunks, autoAcceptThreshold={autoAcceptThreshold}, underReviewThreshold={underReviewThreshold}");
            var (labels, probs) = Predict(data, chunkEmbeddings, trace, neighborCount);

            // Resolve a cluster label to its category. Without a category map, every cluster label is its own category (previous behavior).
            string ResolveCategory(int clusterLabel)
                => categoryMap is not null && categoryMap.TryGetValue(clusterLabel.ToString(), out var mapped)
                    ? mapped
                    : clusterLabel.ToString();

            // Aggregate evidence across all clusters that belong to the same category so a
            // document whose chunks land in different sub-clusters of one category is not diluted.
            var evidence = new Dictionary<string, double>();
            for (int i = 0; i < labels.Length; i++)
            {
                if (labels[i] < 0)
                    continue;
                var category = ResolveCategory(labels[i]);
                evidence[category] = (evidence.TryGetValue(category, out var existing) ? existing : 0) + probs[i];
            }

            if (evidence.Count == 0)
            {
                trace?.Invoke("PredictDocument: evidence is empty (all chunks classified as noise) -> returning (-1, 1.0, under_review)");
                return ("-1", 1.0, "under_review");
            }

            var winner = evidence.OrderByDescending(kv => kv.Value).First();
            var score = winner.Value / chunkEmbeddings.Length;

            trace?.Invoke($"PredictDocument: categoryEvidence=[{string.Join(", ", evidence.Select(kv => $"{kv.Key}:{kv.Value:G4}"))}]");
            trace?.Invoke($"PredictDocument: winner=category {winner.Key}, evidenceSum={winner.Value:G4}, score={score:G4}");

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
                label = winner.Key;
                decision = "auto_classified";
                trace?.Invoke($"PredictDocument: score >= {autoAcceptThreshold} -> label={label}, auto_classified");
            }
            else
            {
                label = winner.Key;
                decision = "under_review";
                trace?.Invoke($"PredictDocument: score in [{underReviewThreshold}, {autoAcceptThreshold}) -> label={label}, under_review");
            }

            return (label, score, decision);
        }

        public static (string label, double score, string decision) PredictKnn(
            PredictionData data,
            double[][] chunkEmbeddings,
            IReadOnlyDictionary<string, string>? categoryMap = null,
            int neighborCount = 5,
            double autoAcceptThreshold = 0.95,
            double underReviewThreshold = 0.70,
            IReadOnlyList<string>? vectorCategories = null,
            Action<string>? trace = null)
        {
            trace?.Invoke($"PredictKnn: {chunkEmbeddings.Length} chunks, neighborCount={neighborCount}, autoAcceptThreshold={autoAcceptThreshold}, underReviewThreshold={underReviewThreshold}");

            // Resolve each training vector's category. When explicit per-vector categories are
            // provided (production stores them at build time), use them directly; otherwise derive
            // via internal label -> cluster map -> category.
            var resolvedCategories = new string[data.RawData.Length];
            for (int i = 0; i < data.RawData.Length; i++)
            {
                if (vectorCategories is not null && i < vectorCategories.Count && !string.IsNullOrEmpty(vectorCategories[i]))
                {
                    resolvedCategories[i] = vectorCategories[i];
                    continue;
                }

                var internalLabel = data.Labels[i];
                if (data.ClusterMap.TryGetValue(internalLabel, out var finalLabel) && finalLabel >= 0 &&
                    categoryMap is not null && categoryMap.TryGetValue(finalLabel.ToString(), out var mapped))
                {
                    resolvedCategories[i] = mapped;
                }
                else
                {
                    resolvedCategories[i] = internalLabel.ToString();
                }
            }

            var evidence = new Dictionary<string, double>(StringComparer.Ordinal);
            var chunkWinner = new List<(string Category, double Weight)>();
            for (int p = 0; p < chunkEmbeddings.Length; p++)
            {
                var point = chunkEmbeddings[p];
                var neighborIdxAndDist = Enumerable.Range(0, data.RawData.Length)
                    .Select(i => (Index: i, Distance: CosineDistance(point, data.RawData[i])))
                    .OrderBy(n => n.Distance)
                    .Take(Math.Min(neighborCount, data.RawData.Length))
                    .ToArray();

                var votes = new Dictionary<string, int>(StringComparer.Ordinal);
                var similarityByCategory = new Dictionary<string, double>(StringComparer.Ordinal);
                foreach (var (idx, distance) in neighborIdxAndDist)
                {
                    var category = resolvedCategories[idx];
                    votes[category] = votes.TryGetValue(category, out var count) ? count + 1 : 1;
                    similarityByCategory[category] = similarityByCategory.TryGetValue(category, out var sim) ? sim + (1.0 - distance) : 1.0 - distance;
                }

                var winner = votes.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).First();
                // Similarity-weighted confidence: fraction of agreeing neighbors times how close
                // those neighbors are. Only a chunk whose top-k neighbors all agree AND are
                // near-identical to the training point approaches 1.0.
                var winnerVoteFraction = (double)winner.Value / Math.Min(neighborCount, data.RawData.Length);
                var winnerAvgSimilarity = similarityByCategory[winner.Key] / winner.Value;
                var weight = winnerVoteFraction * winnerAvgSimilarity;
                evidence[winner.Key] = evidence.TryGetValue(winner.Key, out var existingEvidence) ? existingEvidence + weight : weight;
                chunkWinner.Add((winner.Key, weight));
                trace?.Invoke($"  chunk[{p}]: kNN votes=[{string.Join(", ", votes.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}:{kv.Value}"))}] winner={winner.Key} voteFraction={winnerVoteFraction:G4} avgSim={winnerAvgSimilarity:G4} weight={weight:G4}");
            }

            if (evidence.Count == 0)
            {
                trace?.Invoke("PredictKnn: evidence is empty -> returning (-1, 1.0, under_review)");
                return ("-1", 1.0, "under_review");
            }

            var winnerCategory = evidence.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).First();
            var score = winnerCategory.Value / chunkEmbeddings.Length;

            trace?.Invoke($"PredictKnn: categoryEvidence=[{string.Join(", ", evidence.Select(kv => $"{kv.Key}:{kv.Value:G4}"))}]");
            trace?.Invoke($"PredictKnn: winner=category {winnerCategory.Key}, evidenceSum={winnerCategory.Value:G4}, score={score:G4}");

            string decision;
            if (score < underReviewThreshold)
            {
                decision = "under_review";
                trace?.Invoke($"PredictKnn: score < {underReviewThreshold} -> under_review");
            }
            else if (score >= autoAcceptThreshold)
            {
                decision = "auto_classified";
                trace?.Invoke($"PredictKnn: score >= {autoAcceptThreshold} -> auto_classified");
            }
            else
            {
                decision = "under_review";
                trace?.Invoke($"PredictKnn: score in [{underReviewThreshold}, {autoAcceptThreshold}) -> under_review");
            }

            return (winnerCategory.Key, score, decision);
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
