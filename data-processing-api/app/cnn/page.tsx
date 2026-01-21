"use client"

import { useState } from "react"
import { Brain, Cuboid, Activity, Layers, Play, BarChart2 } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import DatasetSelection from "./components/dataset-selection"
import CNNTraining from "./components/cnn-training"
import PredictionsView from "./components/predictions-view"
import CNNArchitecture from "./components/cnn-architecture"
import CNNComparison from "./components/cnn-comparison-view"

export default function CNNPage() {
    const [activeTab, setActiveTab] = useState("dataset")
    const [selectedDataset, setSelectedDataset] = useState<string>("digits")
    const [isTrained, setIsTrained] = useState(false)

    const handleDatasetSelect = (dataset: string) => {
        setSelectedDataset(dataset)
        setIsTrained(false) // Reset training status when dataset changes
    }

    const handleTrainingComplete = () => {
        setIsTrained(true)
        setActiveTab("predictions")
    }

    return (
        <main className="container mx-auto py-6 px-4 max-w-7xl">
            <div className="mb-8">
                <h1 className="text-3xl font-bold tracking-tight">Convolutional Neural Networks</h1>
                <p className="text-muted-foreground mt-1">
                    Train and visualize CNN models on standard image datasets
                </p>
            </div>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="grid w-full grid-cols-5 mb-6">
                    <TabsTrigger value="dataset" className="gap-2">
                        <Cuboid className="h-4 w-4" />
                        <span className="hidden sm:inline">Dataset</span>
                    </TabsTrigger>
                    <TabsTrigger value="training" className="gap-2">
                        <Activity className="h-4 w-4" />
                        <span className="hidden sm:inline">Training</span>
                    </TabsTrigger>
                    <TabsTrigger value="predictions" disabled={!isTrained} className="gap-2">
                        <Play className="h-4 w-4" />
                        <span className="hidden sm:inline">Predictions</span>
                    </TabsTrigger>
                    <TabsTrigger value="architecture" className="gap-2">
                        <Layers className="h-4 w-4" />
                        <span className="hidden sm:inline">Architecture</span>
                    </TabsTrigger>
                    <TabsTrigger value="comparison" className="gap-2">
                        <BarChart2 className="h-4 w-4" />
                        <span className="hidden sm:inline">Comparison</span>
                    </TabsTrigger>
                </TabsList>

                <TabsContent value="dataset">
                    <Card>
                        <CardHeader>
                            <CardTitle>Dataset Selection</CardTitle>
                            <CardDescription>Choose a dataset to explore and train on</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <DatasetSelection
                                selectedDataset={selectedDataset}
                                onSelect={handleDatasetSelect}
                            />
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="training">
                    <Card>
                        <CardHeader>
                            <CardTitle>Model Training</CardTitle>
                            <CardDescription>Configure and train your CNN model</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <CNNTraining
                                dataset={selectedDataset}
                                onTrainingComplete={handleTrainingComplete}
                            />
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="predictions">
                    <Card>
                        <CardHeader>
                            <CardTitle>Predictions & Analysis</CardTitle>
                            <CardDescription>View model predictions and error analysis</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <PredictionsView dataset={selectedDataset} />
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="architecture">
                    <Card>
                        <CardHeader>
                            <CardTitle>Network Architecture</CardTitle>
                            <CardDescription>Visualize the CNN structure</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <CNNArchitecture dataset={selectedDataset} />
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="comparison">
                    <Card>
                        <CardHeader>
                            <CardTitle>Dataset Comparison</CardTitle>
                            <CardDescription>Compare performance across different datasets</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <CNNComparison />
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </main>
    )
}
