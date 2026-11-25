/** @type {import('next').NextConfig} */
const nextConfig = {
  // Videos served directly from FastAPI (no proxy); images use Next Image Loader
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost", port: "8000" },
      { protocol: "http", hostname: "127.0.0.1", port: "8000" },
    ],
  },
};

export default nextConfig;
