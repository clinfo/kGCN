package org.gcnk.knime.nodes;

import java.io.File;
import java.net.URL;
import org.eclipse.core.runtime.Platform;
import org.eclipse.core.runtime.FileLocator;
import org.eclipse.core.runtime.Path;
import org.knime.core.util.FileUtil;
import org.knime.core.node.NodeLogger;
import org.osgi.framework.Bundle;
import org.osgi.framework.BundleActivator;
import org.osgi.framework.BundleContext;

public class Activator implements BundleActivator {

    private static final NodeLogger LOGGER = NodeLogger.getLogger(Activator.class);
	
	@Override
	public void start(BundleContext context) throws Exception {
		// TODO Auto-generated method stub

	}

	@Override
	public void stop(BundleContext context) throws Exception {
		// TODO Auto-generated method stub

	}

    /**
     * Returns the file contained in the plugin with the given ID.
     *
     * @param symbolicName
     *            ID of the plugin containing the file
     * @param relativePath
     *            File path inside the plugin
     * @return The file
     */
    public static File getFile(final String symbolicName, final String relativePath) {
    	try {
            final Bundle bundle = Platform.getBundle(symbolicName);
            final URL url = FileLocator.find(bundle, new Path(relativePath), null);
            return url != null ? FileUtil.getFileFromURL(FileLocator.toFileURL(url)) : null;
        } catch (final Exception e) {
            LOGGER.debug(e.getMessage(), e);
            return null;
        }
	}
    
}
