"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle } from "lucide-react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

interface ResultDisplayProps {
  result: any
}

export function ResultDisplay({ result }: ResultDisplayProps) {
  if (!result) {
    return null
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Clustering Results</CardTitle>
          <CardDescription>{result.message || "Results of the clustering algorithm"}</CardDescription>
        </CardHeader>
        <CardContent>
          {result.error ? (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{result.error}</AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-6">
              {/* Performance Metrics */}
              {result.performance && (
                <div>
                  <h3 className="text-lg font-medium mb-2">Performance Metrics</h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Metric</TableHead>
                        <TableHead>Value</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {Object.entries(result.performance).map(([metric, value]: [string, any]) => (
                        <TableRow key={metric}>
                          <TableCell className="font-medium">
                            {metric.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                          </TableCell>
                          <TableCell>
                            {value !== null ? (typeof value === "number" ? value.toFixed(4) : value) : "N/A"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}

              {/* Visualization */}
              {result.plot_url && (
                <div>
                  <h3 className="text-lg font-medium mb-2">Visualization</h3>
                  <div className="border rounded-md p-2 bg-white">
                    <img
                      src={`http://localhost:5000${result.plot_url}`}
                      alt="Clustering Visualization"
                      className="mx-auto"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
