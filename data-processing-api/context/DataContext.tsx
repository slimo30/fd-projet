"use client"

import React, { createContext, useContext, useState, ReactNode } from "react"

interface DataContextType {
  dataPath: string | null
  processedPath: string | null
  columns: string[]
  setData: (path: string, cols: string[]) => void
  updateProcessedPath: (path: string, cols?: string[]) => void
  clearData: () => void
  hasData: boolean
}

const DataContext = createContext<DataContextType | undefined>(undefined)

export function DataProvider({ children }: { children: ReactNode }) {
  const [dataPath, setDataPath] = useState<string | null>(null)
  const [processedPath, setProcessedPath] = useState<string | null>(null)
  const [columns, setColumns] = useState<string[]>([])

  const setData = (path: string, cols: string[]) => {
    setDataPath(path)
    setColumns(cols)
  }

  const updateProcessedPath = (path: string, cols?: string[]) => {
    setProcessedPath(path)
    setDataPath(path)
    if (cols) {
      setColumns(cols)
    }
  }

  const clearData = () => {
    setDataPath(null)
    setProcessedPath(null)
    setColumns([])
  }

  return (
    <DataContext.Provider value={{
      dataPath,
      processedPath,
      columns,
      setData,
      updateProcessedPath,
      clearData,
      hasData: !!dataPath
    }}>
      {children}
    </DataContext.Provider>
  )
}

export function useData() {
  const context = useContext(DataContext)
  if (context === undefined) {
    throw new Error("useData must be used within a DataProvider")
  }
  return context
}
