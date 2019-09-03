package org.gcnk.knime.nodes.GCNGraphViewer;

import org.knime.core.node.NodeView;

/**
 * <code>NodeView</code> for the "GCNGraphViewer" Node.
 * Display graphically the contribution of each atom of each test compound in the test set to the predicted value on the compound from the output of the GCNVisualizer
 *
 * @author org.gcnk
 */
public class GCNGraphViewerNodeView extends NodeView<GCNGraphViewerNodeModel> {

    /**
     * Creates a new view.
     * 
     * @param nodeModel The model (class: {@link GCNGraphViewerNodeModel})
     */
    protected GCNGraphViewerNodeView(final GCNGraphViewerNodeModel nodeModel) {
        super(nodeModel);

        // TODO instantiate the components of the view here.

    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void modelChanged() {

        // TODO retrieve the new model from your nodemodel and 
        // update the view.
        GCNGraphViewerNodeModel nodeModel = 
            (GCNGraphViewerNodeModel)getNodeModel();
        assert nodeModel != null;
        
        // be aware of a possibly not executed nodeModel! The data you retrieve
        // from your nodemodel could be null, emtpy, or invalid in any kind.
        
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void onClose() {
    
        // TODO things to do when closing the view
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void onOpen() {

        // TODO things to do when opening the view
    }

}

