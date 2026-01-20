"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getImageUrl } from "@/lib/api";

interface ModelResultsProps {
  results: any;
  algorithm: string;
}

export default function ModelResults({ results, algorithm }: ModelResultsProps) {
  if (!results) {
    return (
      <div className="text-center text-muted-foreground">
        No results available. Please run a model first.
      </div>
    );
  }

  const renderMetrics = () => {
    if (!results.metrics) return null;

    const metrics = results.metrics;

    switch (algorithm) {
      case "knn":
        return (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <MetricCard label="Best K Value" value={metrics.best_k} />
            <MetricCard label="Accuracy" value={`${(metrics.accuracy * 100).toFixed(2)}%`} />
            <MetricCard label="Precision" value={`${(metrics.precision * 100).toFixed(2)}%`} />
            <MetricCard label="Recall" value={`${(metrics.recall * 100).toFixed(2)}%`} />
            <MetricCard label="F1 Score" value={metrics.f1_score.toFixed(4)} />
          </div>
        );

      case "naive-bayes":
        return (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Accuracy" value={`${(metrics.accuracy * 100).toFixed(2)}%`} />
            <MetricCard label="Precision" value={`${(metrics.precision * 100).toFixed(2)}%`} />
            <MetricCard label="Recall" value={`${(metrics.recall * 100).toFixed(2)}%`} />
            <MetricCard label="F1 Score" value={metrics.f1_score.toFixed(4)} />
          </div>
        );

      case "decision-tree":
        return (
          <div className="grid grid-cols-2 gap-4">
            <MetricCard label="Accuracy" value={`${(metrics.accuracy * 100).toFixed(2)}%`} />
            <MetricCard label="Criterion Used" value={metrics.criterion_used} />
          </div>
        );

      case "linear-regression":
        return (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <MetricCard label="MSE" value={metrics.mse.toFixed(4)} />
            <MetricCard label="RMSE" value={metrics.rmse.toFixed(4)} />
            <MetricCard label="R² Score" value={metrics.r2_score.toFixed(4)} />
            <MetricCard label="Intercept" value={metrics.intercept.toFixed(4)} />
            {metrics.coefficients && (
              <Card className="col-span-2">
                <CardHeader>
                  <CardTitle className="text-sm">Coefficients</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xs font-mono overflow-x-auto">
                    {metrics.coefficients.map((coef: number, i: number) => (
                      <div key={i}>Feature {i + 1}: {coef.toFixed(4)}</div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        );

      case "neural-network":
        return (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <MetricCard label="Accuracy" value={`${(metrics.accuracy * 100).toFixed(2)}%`} />
            <MetricCard label="Precision" value={`${(metrics.precision * 100).toFixed(2)}%`} />
            <MetricCard label="Recall" value={`${(metrics.recall * 100).toFixed(2)}%`} />
            <MetricCard label="F1 Score" value={metrics.f1_score.toFixed(4)} />
            <MetricCard label="Iterations" value={metrics.iterations} />
          </div>
        );

      default:
        return (
          <pre className="bg-gray-100 p-4 rounded-md overflow-auto">
            {JSON.stringify(metrics, null, 2)}
          </pre>
        );
    }
  };

  const renderVisualization = () => {
    const plotUrl = results.plot_url || results.metrics?.matrix_plot_url;
    
    if (!plotUrl) return null;

    return (
      <Card>
        <CardHeader>
          <CardTitle>Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex justify-center">
            <img
              src={getImageUrl(plotUrl)}
              alt="Model visualization"
              className="max-w-full h-auto rounded-md border"
              style={{ maxHeight: "500px" }}
              onError={(e) => {
                console.error("Error loading image:", plotUrl);
                e.currentTarget.style.display = 'none';
              }}
            />
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-6">
      {results.success !== undefined && (
        <div className="flex items-center gap-2">
          <Badge variant={results.success ? "default" : "destructive"}>
            {results.success ? "Success" : "Failed"}
          </Badge>
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>{renderMetrics()}</CardContent>
      </Card>

      {renderVisualization()}
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
      </CardContent>
    </Card>
  );
}
