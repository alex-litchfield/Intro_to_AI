        distance = 0
        closestPosition = None
        for position in livingGhostPositionDistributions:
            if distance == 0 or self.distancer.getDistance(pacmanPosition, position.argMax()) < distance:
                closestPosition = position.argMax()
                distance = self.distancer.getDistance(pacmanPosition, position.argMax())

        actionToReturn = None
        for action in legal:
            if self.distancer.getDistance(Actions.getSuccessor(pacmanPosition, action), closestPosition) <= distance:
                actionToReturn = action
                distance = self.distancer.getDistance(Actions.getSuccessor(pacmanPosition, action), closestPosition)
        return actionToReturn
    
        # Step 1: Get most likely positions of all living ghosts
        mostLikelyPositions = [
            dist.argMax() for dist in livingGhostPositionDistributions
        ]

        # Step 2: Find the closest ghost position
        minGhostDistance = float('inf')
        closestGhostPos = None
        for pos in mostLikelyPositions:
            dist = self.distancer.getDistance(pacmanPosition, pos)
            if dist < minGhostDistance:
                minGhostDistance = dist
                closestGhostPos = pos

        # Step 3: Choose the action that minimizes distance to the closest ghost
        bestAction = Directions.STOP
        minActionDistance = float('inf')
        for action in legal:
            successorPos = Actions.getSuccessor(pacmanPosition, action)
            dist = self.distancer.getDistance(successorPos, closestGhostPos)
            if dist < minActionDistance:
                minActionDistance = dist
                bestAction = action

        return bestAction