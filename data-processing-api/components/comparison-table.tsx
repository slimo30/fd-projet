"use client"

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

interface ComparisonTableProps {
  data: Record<string, any>
}

export function ComparisonTable({ data }: ComparisonTableProps) {
  if (!data || Object.keys(data).length === 0) {
    return <div>No comparison data available</div>
  }

  const algorithms = Object.keys(data)
  const metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]

  // Find best algorithm for each metric
  const bestAlgorithm: Record<string, string> = {}

  metrics.forEach((metric) => {
    let bestValue: number | null = null
    let bestAlgo = ""

    algorithms.forEach((algo) => {
      const value = data[algo][metric]

      if (value !== null) {
        if (
          bestValue === null ||
          (metric === "silhouette" && value > bestValue) ||
          (metric === "davies_bouldin" && value < bestValue) ||
          (metric === "calinski_harabasz" && value > bestValue)
        ) {
          bestValue = value
          bestAlgo = algo
        }
      }
    })

    if (bestAlgo) {
      bestAlgorithm[metric] = bestAlgo
    }
  })

  return (
    <div>
      <h3 className="text-lg font-medium mb-2">Algorithm Comparison</h3>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Algorithm</TableHead>
              {metrics.map((metric) => (
                <TableHead key={metric}>{metric.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {algorithms.map((algo) => (
              <TableRow key={algo}>
                <TableCell className="font-medium">{algo.toUpperCase()}</TableCell>
                {metrics.map((metric) => (
                  <TableCell key={`${algo}-${metric}`}>
                    {data[algo][metric] !== null ? (
                      <div className="flex items-center gap-2">
                        {data[algo][metric].toFixed(4)}
                        {bestAlgorithm[metric] === algo && (
                          <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                            Best
                          </Badge>
                        )}
                      </div>
                    ) : (
                      "N/A"
                    )}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      <div className="mt-4 text-sm text-muted-foreground">
        <p>
          <strong>Silhouette Score:</strong> Higher is better. Measures how similar an object is to its own cluster
          compared to other clusters.
        </p>
        <p>
          <strong>Davies-Bouldin Index:</strong> Lower is better. Measures the average similarity between clusters.
        </p>
        <p>
          <strong>Calinski-Harabasz Index:</strong> Higher is better. Measures the ratio of between-cluster dispersion
          to within-cluster dispersion.
        </p>
      </div>
    </div>
  )
}
