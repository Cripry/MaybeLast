import { NextApiRequest, NextApiResponse } from 'next';
import { PrismaClient, TurbineData } from "@prisma/client";
import { parse } from 'querystring';
import { v4 as uuidv4 } from 'uuid'; // Add this line
import fs from 'fs'; // Add this line
import path from 'path'; // Add this line
import { exec } from 'child_process';


const prisma = new PrismaClient();

export const config = {
    api: {
        bodyParser: false,
    },
};



async function getFirst10TurbineData(): Promise<TurbineData[]> {
    try {
        const categories = await prisma.turbineData.findMany({
            take: 10,
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
        // ... (no changes here)
    } else if (req.method === "POST") {
        const data: any = [];
        req.on('data', chunk => {
            data.push(chunk);
        });
        req.on('end', () => {
            const buffer = Buffer.concat(data);
            const fileUuid = uuidv4();
            const fileExtension = '.csv';  // Changed to CSV as that is likely what you are working with
            const fileName = `${fileUuid}${fileExtension}`;
            const uploadPath = path.join(process.cwd(), 'uploads', fileName);

            fs.writeFile(uploadPath, buffer, (err) => {
                if (err) {
                    console.error('Error writing file:', err);
                    return res.status(500).json({ success: false, message: 'Failed to write file' });
                }

                // Remove boundary string
                let fileContent = fs.readFileSync(uploadPath, 'utf-8');
                fileContent = fileContent.replace(/------WebKitFormBoundary[a-zA-Z0-9]*--/g, '');
                fs.writeFileSync(uploadPath, fileContent);

                // Run Python script
                exec(`python3.9 src/scripts/data_processor.py ${uploadPath}`, (error, stdout, stderr) => {
                    if (error) {
                        console.error(`Python script execution error: ${error}`);
                        return res.status(500).json({ success: false, message: 'Python script failed' });
                    }
                    // ... (no changes here)
                    res.status(200).json({ success: true, message: 'File uploaded and processed successfully' });
                });
            });
        });
    } else {
        res.status(405).json({ error: "Method not allowed" });
    }
}