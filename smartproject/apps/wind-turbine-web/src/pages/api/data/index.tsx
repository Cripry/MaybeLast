import { NextApiRequest, NextApiResponse } from 'next';
import { PrismaClient, TurbineData } from "@prisma/client";

const prisma = new PrismaClient();

async function getTurbineData(take: number): Promise<TurbineData[]> {
  try {
    const categories = await prisma.turbineData.findMany({
      take,  // replaced hard-coded '10' with the parameter 'take'
      orderBy: {
        id: 'desc'
      }
    });

    return categories;
  } catch (error) {
    console.error("Error fetching data:", error);
    throw error;
  } finally {
    await prisma.$disconnect();
  }
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === "GET") {
    const take = parseInt(req.query.take as string) || 10;  // if 'take' is not provided, default to 10

    try {
      const data = await getTurbineData(take);
      res.status(200).json(data);
    } catch (error) {
      res.status(500).json({ error: "Internal server error" });
    }
  } else {
    res.status(405).json({ error: "Method not allowed" });
  }
}
