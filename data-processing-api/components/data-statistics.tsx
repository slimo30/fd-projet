"use client";

import { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";

interface DataStatisticsProps {
  path: string;
}

export default function DataStatistics({ path }: DataStatisticsProps) {
  const [statistics, setStatistics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStatistics = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`/api/data/statistics?path=${path}`);

        if (!response.ok) {
          throw new Error(`Failed to fetch statistics: ${response.statusText}`);
        }

        const result = await response.json();
        setStatistics(result || {});
      } catch (error) {
        console.error("Error fetching statistics:", error);
        setError(
          error instanceof Error ? error.message : "An unknown error occurred"
        );
      } finally {
        setLoading(false);
      }
    };

    if (path) {
      fetchStatistics();
    }
  }, [path]);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-40">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 text-red-500 p-4 rounded-md">
        <p>{error}</p>
      </div>
    );
  }

  if (!statistics || Object.keys(statistics).length === 0) {
    return (
      <div className="text-center p-4">
        <p>No statistics available</p>
      </div>
    );
  }

  // Group columns by type
  const numericColumns: string[] = [];
  const categoricalColumns: string[] = [];

  Object.entries(statistics).forEach(([column, stats]: [string, any]) => {
    if (stats.mean !== null) {
      numericColumns.push(column);
    } else {
      categoricalColumns.push(column);
    }
  });

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Columns</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {Object.keys(statistics).length}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">
              Numeric Columns
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{numericColumns.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">
              Categorical Columns
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{categoricalColumns.length}</p>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-4">
        <h3 className="text-lg font-medium">Numeric Columns</h3>
        {numericColumns.length > 0 ? (
          <ScrollArea className="h-[300px] rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Column</TableHead>
                  <TableHead>Mean</TableHead>
                  <TableHead>Median</TableHead>
                  <TableHead>Mode</TableHead>
                  <TableHead>Missing Values</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {numericColumns.map((column) => {
                  const stats = statistics[column];
                  return (
                    <TableRow key={column}>
                      <TableCell className="font-medium">{column}</TableCell>
                      <TableCell>{stats.mean?.toFixed(2) || "N/A"}</TableCell>
                      <TableCell>{stats.median?.toFixed(2) || "N/A"}</TableCell>
                      <TableCell>
                        {stats.mode !== null ? String(stats.mode) : "N/A"}
                      </TableCell>
                      <TableCell>{stats.missing || "N/A"}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </ScrollArea>
        ) : (
          <p className="text-center text-muted-foreground">
            No numeric columns found
          </p>
        )}
      </div>

      <div className="space-y-4">
        <h3 className="text-lg font-medium">Categorical Columns</h3>
        {categoricalColumns.length > 0 ? (
          <ScrollArea className="h-[300px] rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Column</TableHead>
                  <TableHead>Mode</TableHead>
                  <TableHead>Missing Values</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {categoricalColumns.map((column) => {
                  const stats = statistics[column];
                  return (
                    <TableRow key={column}>
                      <TableCell className="font-medium">{column}</TableCell>
                      <TableCell>
                        {stats.mode !== null ? String(stats.mode) : "N/A"}
                      </TableCell>
                      <TableCell>{stats.missing || "N/A"}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </ScrollArea>
        ) : (
          <p className="text-center text-muted-foreground">
            No categorical columns found
          </p>
        )}
      </div>
    </div>
  );
}
