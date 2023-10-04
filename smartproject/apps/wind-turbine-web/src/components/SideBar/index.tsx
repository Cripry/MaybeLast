// apps/wind-turbine/src/components/SideBar/index.tsx
import * as React from 'react';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import InboxIcon from '@mui/icons-material/MoveToInbox';
import MailIcon from '@mui/icons-material/Mail';
import { useRouter } from 'next/router';

const drawerWidth = 240;

interface Props {
	window?: () => Window;
}

export default function SideBar(props: Props) {
	const { window } = props;
	const container = window !== undefined ? () => window().document.body : undefined;
	const router = useRouter();

	const handleNavigation = (href: string) => () => {
		router.push(href);
	};

	const drawer = (
		<div>
			<List>
				{[
					{ text: 'Dashboard', href: '/dashboard' },
					{ text: 'Data', href: '/data' }
				].map((item, index) => (
					<ListItem key={item.text} disablePadding>
						<ListItemButton onClick={handleNavigation(item.href)} sx={{ width: '100%' }}>
							<ListItemIcon>
								{index % 2 === 0 ? <InboxIcon /> : <MailIcon />}
							</ListItemIcon>
							<ListItemText primary={item.text} />
						</ListItemButton>
					</ListItem>
				))}
			</List>
		</div>
	);

	return (
		<Drawer
			variant="permanent"
			open
			sx={{
				'& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
			}}
		>
			{drawer}
		</Drawer>
	);
}
