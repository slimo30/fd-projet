import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    domains: ['127.0.0.1', 'localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/static/:path*',
        destination: 'http://127.0.0.1:5001/static/:path*',
      },
      {
        source: '/ml/:path*',
        destination: 'http://127.0.0.1:5001/ml/:path*',
      },
      {
        source: '/api/ml/:path*',
        destination: 'http://127.0.0.1:5001/ml/:path*',
      },
      {
        source: '/clustering/:path*',
        destination: 'http://127.0.0.1:5001/clustering/:path*',
      },
      {
        source: '/upload-data-clustering',
        destination: 'http://127.0.0.1:5001/upload-data-clustering',
      },
      {
        source: '/plot/:path*',
        destination: 'http://127.0.0.1:5001/plot/:path*',
      },
      {
        source: '/create-sample',
        destination: 'http://127.0.0.1:5001/create-sample',
      },
      {
        source: '/data/:path*',
        destination: 'http://127.0.0.1:5001/data/:path*',
      },
      {
        source: '/cnn/:path*',
        destination: 'http://127.0.0.1:5001/cnn/:path*',
      },
    ];
  },
};

export default nextConfig;
