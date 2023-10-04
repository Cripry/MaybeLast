// apps/wind-turbine/src/pages/_app.tsx
import type { AppProps } from 'next/app';
import { useEffect } from 'react';
import { useRouter } from 'next/router';
import SideBar from '../components/SideBar';

function MyApp({ Component, pageProps }: AppProps) {
    const router = useRouter();

    useEffect(() => {
        if (router.pathname === '/') {
            router.replace('/dashboard');
        }
    }, [router]);

    return (
        <div style={{ display: 'flex' }}>
            <SideBar />
            <div style={{ marginLeft: 240 }}>  {/* matching drawerWidth */}
                <Component {...pageProps} />
            </div>
        </div>
    )
}

export default MyApp;
