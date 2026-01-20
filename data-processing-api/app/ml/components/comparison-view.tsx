"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Loader2, AlertCircle, RefreshCw } from "lucide-react";
import { mlApi, ApiError } from "@/lib/api";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function ComparisonView() {
  const [comparison, setComparison] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchComparison = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await mlApi.getComparison();
      setComparison(data);
    } catch (err: any) {
      const errorMessage = err instanceof ApiError ? err.message : (err.message || "Failed to load comparison");
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchComparison();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-40">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          {error}
          <Button variant="outline" size="sm" className="mt-2" onClick={fetchComparison}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  if (!comparison || Object.keys(comparison).length === 0) {
    return (
      <div className="text-center text-muted-foreground p-8">
        <p className="mb-4">No algorithms have been run yet.</p>
        <p className="text-sm">Run some ML algorithms to see a comparison of their performance.</p>
      </div>
    );
  }

  const algorithms = Object.keys(comparison);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Algorithm Performance Comparison</h3>
        <Button variant="outline" size="sm" onClick={fetchComparison}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="font-bold">Algorithm</TableHead>
              <TableHead className="text-right">Accuracy</TableHead>
              <TableHead className="text-right">Precision</TableHead>
              <TableHead className="text-right">Recall</TableHead>
              <TableHead className="text-right">F1 Score</TableHead>
              <TableHead className="text-right">Additional Metrics</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {algorithms.map((algo) => {
              const metrics = comparison[algo].metrics || comparison[algo];
              
              return (
                <TableRow key={algo}>
                  <TableCell className="font-medium capitalize">
                    {algo.replace(/-/g, ' ')}
                  </TableCell>
                  <TableCell className="text-right">
                    {metrics.accuracy !== undefined
                      ? `${(metrics.accuracy * 100).toFixed(2)}%`
                      : 'N/A'}
                  </TableCell>
                  <TableCell className="text-right">
                    {metrics.precision !== undefined
                      ? `${(metrics.precision * 100).toFixed(2)}%`
                      : 'N/A'}
                  </TableCell>
                  <TableCell className="text-right">
                    {metrics.recall !== undefined
                      ? `${(metrics.recall * 100).toFixed(2)}%`
                      : 'N/A'}
                  </TableCell>
                  <TableCell className="text-right">
                    {metrics.f1_score !== undefined
                      ? metrics.f1_score.toFixed(4)
                      : 'N/A'}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="text-xs space-y-1">
                      {metrics.best_k && <div>Best K: {metrics.best_k}</div>}
                      {metrics.mse && <div>MSE: {metrics.mse.toFixed(4)}</div>}
                      {metrics.r2_score && <div>R²: {metrics.r2_score.toFixed(4)}</div>}
                      {metrics.criterion_used && <div>Criterion: {metrics.criterion_used}</div>}
                      {metrics.iterations && <div>Iterations: {metrics.iterations}</div>}
                    </div>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {algorithms.map((algo) => {
          const result = comparison[algo];
          const metrics = result.metrics || result;
          
          return (
            <Card key={algo}>
              <CardHeader>
                <CardTitle className="text-base capitalize">
                  {algo.replace(/-/g, ' ')}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="space-y-1 text-sm">
                  {metrics.accuracy !== undefined && (
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Accuracy:</dt>
                      <dd className="font-medium">{(metrics.accuracy * 100).toFixed(2)}%</dd>
                    </div>
                  )}
                  {metrics.precision !== undefined && (
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Precision:</dt>
                      <dd className="font-medium">{(metrics.precision * 100).toFixed(2)}%</dd>
                    </div>
                  )}
                  {metrics.recall !== undefined && (
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Recall:</dt>
                      <dd className="font-medium">{(metrics.recall * 100).toFixed(2)}%</dd>
                    </div>
                  )}
                  {metrics.f1_score !== undefined && (
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">F1 Score:</dt>
                      <dd className="font-medium">{metrics.f1_score.toFixed(4)}</dd>
                    </div>
                  )}
                </dl>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
