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
    ];
  },
};

export default nextConfig;
